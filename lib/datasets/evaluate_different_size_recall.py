import imdb
from pascal_voc import pascal_voc
from voc_eval import *
import os, sys
import ipdb
import cPickle
import numpy as np

class pascal_voc_enhanced( pascal_voc ):
    def __init__( self, image_set, year, devkit_path=None ):
        pascal_voc.__init__( self, image_set, year, devkit_path )

    def evaluate_detections_different_size_box(self, all_boxes, output_dir):
        self.evaluate_recall_specific_size( all_boxes , 0.5 , 'small')

    def evaluate_recall_specific_size( self, all_boxes , ovthresh = 0.5 , area='all' ):

        cache_file = os.path.join( self.cache_path , 'voc_' + self._year + self._image_set + 'gt_annotation.pkl' )
        imageSetFile   = os.path.join( self._devkit_path , 'VOC' + self._year , 'ImageSets' , 'Main' , self._image_set + '.txt')
        assert os.path.exists( imageSetFile ) , " the image file doesn't exits "

        with open( imageSetFile , 'r' ) as f:
            lines = f.readlines()
        imageNames = [ x.strip() for x in lines ]

        if os.path.isfile( cache_file ):
            # if the file is cached, then we read from the cache to save time
            with open( cache_file , 'r' ) as f:
                recs = cPickle.load(f)
        else:
            #if the file doesn't exits, then read from annotation file
            recs = {}
            AnnotationFile = os.path.join( self._devkit_path , 'VOC' + self._year , 'Annotations' , '{:s}.xml' )

            for i, img_name in enumerate( imageNames ):
                assert os.path.exists( AnnotationFile.format(img_name) ) , " the xml annotation file doesn't exits "
                recs[img_name] = parse_rec( AnnotationFile.format(img_name) )
                print "reading xml from annotation dir: {:d}/{:d} ".format( i+1 , len(imageNames) )

            print "saving cache file for annotation xml"
            with open( cache_file , 'w' ) as f:
                cPickle.dump( recs , f )

        #define several different sizes
	areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
		  '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}

	area_ranges = [ [0**2, 1e5**2],    # all
			[0**2, 32**2],     # small
			[32**2, 96**2],    # medium
			[96**2, 1e5**2],   # large
			[96**2, 128**2],   # 96-128
			[128**2, 256**2],  # 128-256
			[256**2, 512**2],  # 256-512
			[512**2, 1e5**2],  # 512-inf
		      ]
	assert areas.has_key(area), 'unknown area range: {}'.format(area)
	area_range = area_ranges[areas[area]]

        # extract the boxes coincidence with the specific size

        recs_right_size = {}
        npos = 0

        for img_name in imageNames:
            # get rid of every annotation box doesn't fit the size condition
            R = []
            for obj in recs[img_name]:
                bbox = np.array(obj['bbox']).astype(float)
                area = ( bbox[2] - bbox[0] + 1. )  * ( bbox[3] - bbox[1] + 1. )
                if area >= area_range[0] and area < area_range[1]:
                    R.append(obj)

            bbox = np.array( [ x['bbox'] for x in R ] )
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            cls = np.array( [ self._class_to_ind[x['name']] for x in R ] )

            recs_right_size[img_name] = { 'bbox' : bbox, 'difficult' :difficult,  'det' : det , 'cls' : cls }

        # convert all_boxes array into the following form
        # [ image_index , cls , score , x1 , y1 , x2 , y2 ]
        # the form above will be separated into 4 variables

        # each box's image index
        bbox_img_ids = []
        # each box's label in int format
        bbox_cls = []
        # each box's confidence
        bbox_score = []
        # each box's position in [x1, y1, x2, y2] format
        bbox_xy_pos = []

        for cls_ind , cls in enumerate( self.classes ):
            if cls == '__background__':
                continue
            for im_ind , index in enumerate( self.image_index ):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue

                for k in xrange( dets.shape[0] ):
                    bbox_img_ids.append( index )
                    bbox_cls.append( cls_ind )
                    bbox_score.append( dets[k,-1] )
                    # plus 1 following VOC 1 based manner
                    bbox_xy_pos.append( [dets[k,0]+1 , dets[k,1]+1 , dets[k,2]+1 , dets[k,3]+1 ] )

        bbox_score  = np.array( bbox_score ).astype( float )
        bbox_xy_pos = np.array( bbox_xy_pos).astype( float )

        # sort the detections by score from high to low
        sorted_ind = np.argsort( -bbox_score )

        bbox_score = -np.sort( -bbox_score )
        bbox_img_ids = [ bbox_img_ids[x] for x in sorted_ind ]
        bbox_cls     = [ bbox_cls[x] for x in sorted_ind ]
        bbox_xy_pos  = bbox_xy_pos[sorted_ind , :]

        nd = len( bbox_img_ids )
        
        print 'we have detected {:d} annotations in all'.format(nd)
        tp = np.zeros(nd)

        for d in xrange(nd):
            R = recs_right_size[ bbox_img_ids[d] ]
            ind_right_cls = np.where( R['cls'] == bbox_cls[d] )[0]

            #difficult = R['difficult'][ind_right_cls]
            #det = [ R['det'][x] for x in ind_right_cls ]

            bb = bbox_xy_pos[d,:].astype(float)
            ovmax = -np.inf
            
            if len( ind_right_cls ) > 0:
                BBGT = R['bbox'][ind_right_cls , :]

                ixmin = np.maximum( BBGT[:,0] , bb[0] )
                iymin = np.maximum( BBGT[:,1] , bb[1] )
                ixmax = np.minimum( BBGT[:,2] , bb[2] )
                iymax = np.minimum( BBGT[:,3] , bb[3] )

                iw = np.maximum( ixmax - ixmin +1. , 0. )
                ih = np.maximum( iymax - iymin +1. , 0. )

                inters = iw * ih

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max( overlaps )
                jmax = np.argmax( overlaps )

            if ovmax > ovthresh:
                jindex = ind_right_cls[jmax]
                if not R['difficult'][jindex]:
                    if not R['det'][jindex]:
                        tp[d] = 1.
                        R['det'][jindex] = 1

        tp = np.cumsum(tp)
        recall = tp/float( npos )

        print "number of qualified boxes is:{:d}".format(npos)
        print 'mean of true positive is {:f}'.format( recall.mean() )




"""
	gt_overlaps = np.zeros(0)
        ipdb.set_trace()

	num_pos = 0
	for i in xrange(self.num_images):
	    # Checking for max_overlaps == 1 avoids including crowd annotations
	    # (...pretty hacking :/)
	    max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
	    gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
			       (max_gt_overlaps == 1))[0]
	    gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
	    gt_areas = self.roidb[i]['seg_areas'][gt_inds]
	    valid_gt_inds = np.where((gt_areas >= area_range[0]) &
				     (gt_areas <= area_range[1]))[0]
	    gt_boxes = gt_boxes[valid_gt_inds, :]
	    num_pos += len(valid_gt_inds)

	    if candidate_boxes is None:
		# If candidate_boxes is not supplied, the default is to use the
		# non-ground-truth boxes from this roidb
		non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
		boxes = self.roidb[i]['boxes'][non_gt_inds, :]
	    else:
		boxes = candidate_boxes[i]
	    if boxes.shape[0] == 0:
		continue
	    if limit is not None and boxes.shape[0] > limit:
		boxes = boxes[:limit, :]

	    overlaps = bbox_overlaps(boxes.astype(np.float),
				     gt_boxes.astype(np.float))

	    _gt_overlaps = np.zeros((gt_boxes.shape[0]))
	    for j in xrange(gt_boxes.shape[0]):
		# find which proposal box maximally covers each gt box
		argmax_overlaps = overlaps.argmax(axis=0)
		# and get the iou amount of coverage for each gt box
		max_overlaps = overlaps.max(axis=0)
		# find which gt box is 'best' covered (i.e. 'best' = most iou)
		gt_ind = max_overlaps.argmax()
		gt_ovr = max_overlaps.max()
		assert(gt_ovr >= 0)
		# find the proposal box that covers the best covered gt box
		box_ind = argmax_overlaps[gt_ind]
		# record the iou coverage of this gt box
		_gt_overlaps[j] = overlaps[box_ind, gt_ind]
		assert(_gt_overlaps[j] == gt_ovr)
		# mark the proposal box and the gt box as used
		overlaps[box_ind, :] = -1
		overlaps[:, gt_ind] = -1
	    # append recorded iou coverage level
	    gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

	gt_overlaps = np.sort(gt_overlaps)
	if thresholds is None:
	    step = 0.05
	    thresholds = np.arange(0.5, 0.95 + 1e-5, step)
	recalls = np.zeros_like(thresholds)
	# compute recall for each iou threshold
	for i, t in enumerate(thresholds):
	    recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
	# ar = 2 * np.trapz(recalls, thresholds)
	ar = recalls.mean()
	return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
		'gt_overlaps': gt_overlaps}
                """




def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])


    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
