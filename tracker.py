import logging
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    logger = logging.getLogger('ei').getChild(__name__)
    """
    site: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    @author: Adrian_Rosebrock
    @contact: adrian@pyimagesearch.com
    @copyright: 2018 PyImageSearch. All Rights Reserved.
    @date: 30/12/2018
    Modified by Julian Schwanbeck
    """

    def __init__(self, max_disappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.additional_info = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = max_disappeared

    def register(self, centroid, additional_info):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.additional_info[self.nextObjectID] = additional_info
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, object_id):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[object_id]
        del self.additional_info[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # check to see if the list_i of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in self.disappeared.keys():
                self.disappeared[object_id] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.deregister(object_id)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        # initialize an array of input centroids for the current frame
        # rects should look like [((x, y, ...), (additional_info)), ((x, y, ...), (additional_info)), ...]
        input_centroids = np.zeros((len(rects), len(rects[0][0])), dtype="float")  # length x dimension(s)
        # Here length is amount of data points, dimensions is width of tuples passed by rects (x, y, [, illumination])

        # Initialise additional_info_dict to store additional information per centroid
        additional_info_dict = {}
        # Original version:
        # loop over the bounding box rectangles
        # for (i, (startX, startY, endX, endY)) in enumerate(rects):  # Original
        #     use the bounding box coordinates to derive the centroid
        #     cX = int((startX + endX) / 2.0)
        #     cY = int((startY + endY) / 2.0)
        # input_centroids[i] = (cX, cY)

        # modified for cv2.boundingRect() values
        # x/y(/illumination) coordinates of cv2.boundingRect() is already centroid
        for i, (coords, additional_info) in enumerate(rects):
            input_centroids[i] = coords
            # Everything else goes here with the same key so we can find it later again
            additional_info_dict[i] = additional_info

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], additional_info_dict[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            distance_matrix = dist.cdist(np.array(object_centroids), input_centroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list_i
            rows = distance_matrix.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list_i
            cols = distance_matrix.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in used_rows or col in used_cols:
                    # continue jumps back to the beginning of the for-loop
                    continue
                # @todo: distance check? Currently kicked out in find_good_tracks()/like_a_record_baby.py
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.additional_info[object_id] = additional_info_dict[col]
                self.disappeared[object_id] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

            # compute both the row and column index we have NOT yet examined
            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants de-registering the object
                    if self.disappeared[object_id] > self.maxDisappeared:
                        self.deregister(object_id)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], additional_info_dict[col])
            # return the set of trackable objects
        return self.objects, self.additional_info
