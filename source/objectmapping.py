import numpy as np
import operator
from PIL import Image, ImageDraw, ImageFont
from matplotlib.pyplot import imshow
from itertools import combinations, product
from string import ascii_uppercase

class ObjectMapping:
    '''
        Required:
        import numpy as np
        import operator
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib.pyplot import imshow
        from itertools import combinations, product
        from string import ascii_uppercase
    ''' 
    
    def __init__ (self, filename, results, class_names):
        self.filename = filename
        self.r = results                    # results contain rois, class_ids, masks, and scores
        self.class_names = class_names
        self.img_height = self.r['masks'].shape[0]
        self.img_width = self.r['masks'].shape[1]
        self.total_objects = len(self.r['rois'])
    
    def get_box(self, object_id):
        object_id = object_id-1
        h1 = self.r['rois'][object_id][0]
        h2 = self.r['rois'][object_id][2]
        w1 = self.r['rois'][object_id][1]
        w2 = self.r['rois'][object_id][3]
        return (h1, w1, h2, w2)
    
    def get_objectID(self):
        return {index:self.class_names[self.r['class_ids'][index-1]] for index, array in enumerate(self.r['rois'],\
                                                                                                   start=1)}
    
    def object_class(self, object_id):
        object_id = object_id-1
        return self.class_names[self.r['class_ids'][object_id]]
    
    def count_objects(self):
        "summarize type of objects detected with count"
        objects = [self.class_names[index] for index in self.r['class_ids']]
        objects = dict(zip(*np.unique(objects, return_counts=True)))
        return objects
    
    def get_mask(self, object_id):
        object_id = object_id-1
        return self.r['masks'][:,:,object_id]
    
    def _merge_masks(self, *args):
        """Internal. Merge mask boolean arrays"""
        mask = self._false_canvas()
        for ids in args:
            if(isinstance(ids, np.ndarray)):
                mask = np.bitwise_or(mask, ids.copy())
            else:    
                mask = np.bitwise_or(mask, self.get_mask(ids).copy())
        return mask
    
    def _show_massbox(self, *args, size=2):
        mass_boxes = self._false_canvas()
        """Internal. Only for displaying mass boxes for masks that have an object ID"""
        for ids in args:
            h1, w1, h2, w2 = self.mass_box(ids)
            mass_boxes[h1:h2, w1:w2] = True
            mass_boxes[h1+size:h2-size, w1+size:w2-size] = False
        return mass_boxes
    
    def show_mask(self, *args, show_massbox = False):
        """Creates PIL image from a matrix of booleans. Shows a mask that is either
           directly passed as a boolean matrix or that is retrieved using the object ID.
           Mass box is only for a mask that is retrieved with the object ID."""
        mask = self._merge_masks(*args)
        if show_massbox:
            mass_boxes = self._show_massbox(*args)
            mask = np.bitwise_or(mask, mass_boxes)
        mask_size = mask.shape[::-1]
        maskbytes = np.packbits(mask, axis=1)
        return Image.frombytes(mode='1', size=mask_size, data=maskbytes)
           
    def box_center(self, object_id):
        h1, w1, h2, w2 = self.get_box(object_id)
        hbb_center = int((h1+h2)/2)
        wbb_center = int((w1+w2)/2)
        return (hbb_center, wbb_center)

    def mask_pixel_count(self, object_id):
        h1, w1, h2, w2 = self.get_box(object_id)
        mask = self.get_mask(object_id)
        return (np.sum(mask[h1:h2, w1:w2]))
    
    def mask_pixel_count(self, object_id, h1, w1, h2, w2):
        mask = self.get_mask(object_id)
        return (np.sum(mask[h1:h2, w1:w2]))
    
    def _best_coord(self, object_id, current_coords, step_coord, add=True):
        """Internal only. As edges of the bounding box are scanned in one at a time,
           this returns the coordinate that maximizes number of mask pixels multiplied 
           by the percentage of mask pixels remaining in the moving bounding box."""
        step=1
        step_variable = current_coords[step_coord]
        h1, w1, h2, w2 = current_coords
        bmask = self.get_mask(object_id)
        true_count = np.sum(bmask[h1:h2, w1:w2])
        bmask_area = bmask.shape[0]*bmask.shape[1]
        
        check_max = (true_count/bmask_area)*true_count # Track largest product of perc and count
        temp_area = bmask_area                         # Initialize to any value. Shrinks as step_variable changes
        while(temp_area != 0):
            if(add):
                step_variable = step_variable + step
            else:
                step_variable = step_variable - step
                
            box_adj = {0:bmask[step_variable:h2, w1:w2],
                       1:bmask[h1:h2, step_variable:w2],
                       2:bmask[h1:step_variable, w1:w2],
                       3:bmask[h1:h2, w1:step_variable]}
            
            temp_mask = box_adj[step_coord]
            temp_true = np.sum(temp_mask)
            temp_area = temp_mask.shape[0]*temp_mask.shape[1]
            if (temp_area != 0):
                temp_perc = temp_true/temp_area
            if (temp_true*temp_perc > check_max):
                best_step_variable = step_variable
                check_max = temp_true*temp_perc       
        return best_step_variable
    
    def mass_box(self, object_id):
        """Adjustment to bounding box to reflect a better center of mass"""
        h1, w1, h2, w2 = self.get_box(object_id)
        w1_best = self._best_coord(object_id, (h1, w1, h2, w2), 1, add=True)
        w2_best = self._best_coord(object_id, (h1, w1_best, h2, w2), 3, add=False)
        h2_best = self._best_coord(object_id, (h1, w1_best, h2, w2_best), 2, add=False)
        h1_best = self._best_coord(object_id, (h1, w1_best, h2_best, w2_best), 0, add=True)
                             
        return(h1_best, w1_best, h2_best, w2_best)
                                
    def mass_center(self, object_id):
        h1, w1, h2, w2 = self.mass_box(object_id)
        hm_center = int((h1+h2)/2)
        wm_center = int((w1+w2)/2)
        return (hm_center, wm_center)
    
    def object_location(self, object_id):
        imgH_center_range = [0.333*self.img_height, 0.667*self.img_height]
        imgW_center_range = [0.4*self.img_width, 0.6*self.img_width]
        # section canvas into horizontal and vertical thirds
        htop = (0, 0, int(imgH_center_range[0]), self.img_width)
        hcenter = (int(imgH_center_range[0]), 0, int(imgH_center_range[1]), self.img_width)
        hbottom = (int(imgH_center_range[1]), 0, self.img_height, self.img_width)
        wleft = (0, 0, self.img_height, int(imgW_center_range[0]))
        wcenter = (0, int(imgW_center_range[0]), self.img_height, int(imgW_center_range[1]))
        wright = (0, int(imgW_center_range[1]), self.img_height, self.img_width)
        
        # count the number of pixels in each section
        htop_pixels = self.mask_pixel_count(object_id, *htop)
        hcenter_pixels = self.mask_pixel_count(object_id, *hcenter)
        hbottom_pixels = self.mask_pixel_count(object_id, *hbottom)
        wleft_pixels = self.mask_pixel_count(object_id, *wleft)
        wcenter_pixels = self.mask_pixel_count(object_id, *wcenter)
        wright_pixels = self.mask_pixel_count(object_id, *wright)
        
        hloc_dict = {'top':htop_pixels, 'center':hcenter_pixels, 'bottom':hbottom_pixels}
        wloc_dict = {'left':wleft_pixels, 'center':wcenter_pixels, 'right':wright_pixels}
        
        # return the key with the largest value in each dictionary
        hloc = max(hloc_dict.items(), key=operator.itemgetter(1))[0]
        wloc = max(wloc_dict.items(), key=operator.itemgetter(1))[0]
        return (hloc, wloc)
    
    def _edge_pixels(self, object_id, h1, w1, h2, w2, return_true = True):
        """Internal. Returns list of pixels at the True/False border of a mask.
           return_true determines if the list is the coords True or False pixels at border."""
        if(isinstance(object_id, np.ndarray)):
            mask = object_id
        else:
            mask = self.get_mask(object_id)
        edge_pixels = []
        # Scan horizontally to find edge
        for i in range(h1,h2):    
            for j in range(w1,w2):
                if((mask[i, j] != mask[i, j+1]) and (mask[i,j] == False)):
                    if return_true:
                        edge_pixels.append((i,j+1))
                    else:
                        edge_pixels.append((i,j))
                        
                if((mask[i, j] != mask[i, j+1]) and (mask[i,j] == True)):
                    if return_true:
                        edge_pixels.append((i,j))
                    else:
                        edge_pixels.append((i,j+1))

        # Scan vertically to find edge
        for j in range(w1,w2):
            for i in range(h1,h2):
                if((mask[i, j] != mask[i+1, j]) and (mask[i, j] == False)):
                    if return_true:
                        edge_pixels.append((i+1,j))
                    else:
                        edge_pixels.append((i,j))
                if((mask[i, j] != mask[i+1, j]) and (mask[i, j] == True)):
                    if return_true:
                        edge_pixels.append((i,j))
                    else:
                        edge_pixels.append((i+1,j))
        return edge_pixels
    
    def inflate_mask(self, object_id, inflation_factor=1):
        h1, w1, h2, w2 = self.get_box(object_id)
        # Inflate box holding the mask but don't extend beyond boundaries of image
        if (h1-inflation_factor >= 0):
            h1 = h1-inflation_factor
        if (w1-inflation_factor >= 0):
            w1 = w1-inflation_factor
        if (h2+inflation_factor <= self.img_height):
            h2 = h2 + inflation_factor
        if (w2+inflation_factor <= self.img_width):
            w2 = w2 + inflation_factor
            
        mask = self.get_mask(object_id).copy()      
        for expand in range(inflation_factor):
            edge_pixels = self._edge_pixels(mask, h1, w1, h2, w2, return_true=False)
            for coords in edge_pixels:
                i, j = coords
                mask[i,j] = True
        return mask
    
    def _false_canvas(self):
        """Internal"""
        return np.full((self.img_height, self.img_width), False, dtype=bool)
    
    def create_box_mask(self, h1, w1, h2, w2):
        false_canvas = self._false_canvas()
        false_canvas[h1:h2, w1:w2] = True
        return false_canvas
    
    def object_outline(self, *args, pad=2):
        """Returns a boolean array of the object outline. Must use show_mask() to view"""
        outline = self._false_canvas()
        for obj in args:
            h1, w1, h2, w2 = self.get_box(obj)
            # Pad but don't extend beyond boundaries of image
            if (h1-pad >= 0):
                h1 = h1-pad
            if (w1-pad >= 0):
                w1 = w1-pad
            if (h2+pad <= self.img_height):
                h2 = h2 + pad
            if (w2+pad <= self.img_width):
                w2 = w2 + pad

            edge_pixels = self._edge_pixels(obj, h1, w1, h2, w2, return_true=True)
            for coords in edge_pixels:
                i,j = coords
                outline[i,j] = True
        return outline
        
    def object_topline(self, *args, pad=2):
        """Returns a boolean array of the object topline. Must use show_mask() to view"""
        topline = self._false_canvas()
        for obj in args:
            h1, w1, h2, w2 = self.get_box(obj)
            # Pad but don't extend beyond boundaries of image
            if (h1-pad >= 0):
                h1 = h1-pad
            if (w1-pad >= 0):
                w1 = w1-pad
            if (h2+pad <= self.img_height):
                h2 = h2 + pad
            if (w2+pad <= self.img_width):
                w2 = w2 + pad

            mask = self.get_mask(obj)
            top_pixels = []
            for j in range(w1,w2):
                for i in range(h1,h2):
                    if((mask[i, j] != mask[i+1, j]) and (mask[i, j] == False)):
                            top_pixels.append((i+1,j))
            for coords in top_pixels:
                i,j = coords
                topline[i,j] = True
        return topline
        
    def object_relations(self, tol=0.1):
        if(self.total_objects <= 1):
            print('Not enough objects detected.')
        else:
            ids = range(1, self.total_objects+1)
            combos = combinations(ids, r=2) # all possible combinations of pairs
            object_relations = {'object relations': {'next to':[], 'above':[], 'below':[],
                                                     'touching':[], 'on':[], 'in':[]}
                               }
            for rel in combos:
                print(f"Analyzing object_id {rel[0]}: {self.object_class(rel[0])} "
                      f"and object_id {rel[1]}: {self.object_class(rel[1])}")
                obja, objb = rel
                flip = rel[::-1]
                h1a, w1a, h2a, w2a = self.get_box(obja)
                h1b, w1b, h2b, w2b = self.get_box(objb)
                # Widen width of box size by tol
                if(w1a-tol*w1a >= 0):
                    w1a_mod = int(w1a-tol*w1a)
                if(w2a+tol*w2a <= self.img_width):
                    w2a_mod = int(w2a+tol*w2a)
                if(w1b-tol*w1b >= 0):
                    w1b_mod = int(w1b-tol*w1b)
                if(w2a+tol*w2a <= self.img_width):
                    w2b_mod = int(w2b+tol*w2b)
  
                maska = self.get_mask(obja).copy()
                maskb = self.get_mask(objb).copy()
                boxa = self.create_box_mask(h1a, w1a_mod, h2a, w2a_mod)
                boxb = self.create_box_mask(h1b, w1b_mod, h2b, w2b_mod)
                h1ma, w1ma, h2ma, w2ma = self.mass_box(obja)
                h1mb, w1mb, h2mb, w2mb = self.mass_box(objb)
                hcentera, wcentera = self.mass_center(obja)
                hcenterb, wcenterb = self.mass_center(objb)
                toplinea = self.object_topline(obja)
                toplineb = self.object_topline(objb)
                
                # boolean position checks
                obj_grounded = np.allclose(h2a, h2b, atol=15)
                touching = np.any(np.bitwise_and(self.inflate_mask(obja), self.inflate_mask(objb)))
                a_on_b = np.any(np.bitwise_and(maska, toplineb)) 
                b_on_a = np.any(np.bitwise_and(maskb, toplinea))
                a_align_b = b_align_a = wcentera in list(range(w1b, w2b)) or wcenterb in list(range(w1a, w2a))
                a_above_b = hcentera < hcenterb
                b_above_a = hcenterb < hcentera
                a_below_b = hcentera > hcenterb
                b_below_a = hcenterb > hcentera
                a_in_b = set(range(h1ma, h2ma)).issubset(set(range(h1mb, h2mb)))\
                         and set(range(w1ma, w2ma)).issubset(set(range(w1mb, w2mb)))
                b_in_a = set(range(h1mb, h2mb)).issubset(set(range(h1ma, h2ma)))\
                         and set(range(w1mb, w2mb)).issubset(set(range(w1ma, w2ma))) 
                        
                if(touching):
                    object_relations['object relations']['touching'].append(rel)
                    if(a_on_b and not obj_grounded and a_above_b and not a_in_b):
                        object_relations['object relations']['on'].append(rel)
                        object_relations['object relations']['above'].append(rel)
                        object_relations['object relations']['below'].append(flip)
                    elif(b_on_a and not obj_grounded and b_above_a and not b_in_a):
                        object_relations['object relations']['on'].append(flip)
                        object_relations['object relations']['above'].append(flip)
                        object_relations['object relations']['below'].append(rel)
                    elif(a_in_b):
                        object_relations['object relations']['in'].append(rel)
                    elif(b_in_a):
                        object_relations['object relations']['in'].append(flip)
                    else:
                        object_relations['object relations']['next to'].append(rel)
                        
                    if(a_above_b and a_align_b):
                        object_relations['object relations']['above'].append(rel)
                    elif(a_below_b and a_align_b):
                        object_relations['object relations']['below'].append(rel)
                        
                    if(b_above_a and b_align_a):
                        object_relations['object relations']['above'].append(flip)    
                    elif(b_below_a and b_align_a):
                        object_relations['object relations']['below'].append(flip)
                        
                else:
                    if(np.any(np.bitwise_and(maska, boxb)) or np.any(np.bitwise_and(maskb, boxa))):
                        object_relations['object relations']['next to'].append(rel)
                    if(a_above_b and a_align_b):
                        object_relations['object relations']['above'].append(rel)
                    elif(b_above_a and b_align_a):
                        object_relations['object relations']['above'].append(flip)
                    elif(a_below_b and a_align_b):
                        object_relations['object relations']['below'].append(rel)
                    elif(b_below_a and b_align_a):
                        object_relations['object relations']['below'].append(flip)
                                              
        return object_relations
    
    def grid_coords(self, object_id, height=3, width=3, grid=False):
        """Get grid coordinates using the bounding box in form 'A1' where 'A1' is the top left grid. If grid = True returns two variables."""
        h1, w1, h2, w2 = self.get_box(object_id)
        letters = ascii_uppercase[0:height]
        numbers = (range(1,width+1))
        combo_labels = product(letters, numbers)
        height_array = np.arange(0, self.img_height, self.img_height/height).astype(int)
        width_array = np.arange(0, self.img_width, self.img_width/width).astype(int)
        combo_coords = product(height_array, width_array)
        label_dict = {k:v for k,v in zip(combo_coords, combo_labels)}
        height_array = np.append(height_array, self.img_height)
        width_array = np.append(width_array, self.img_width)
       
        # align to grid coordinates
        h1_array = h1 < height_array
        h2_array = h2 < height_array
        w1_array = w1 < width_array
        w2_array = w2 < width_array
        
        for i in range(len(height_array)-1):
            if(h1_array[i] != h1_array[i+1]):
                h1_index = i
            if(h2_array[i] != h2_array[i+1]):
                h2_index = i
            if(w1_array[i] != w1_array[i+1]):
                w1_index = i
            if(w2_array[i] != w2_array[i+1]):
                w2_index = i
                
        h_align = height_array[h1_index:h2_index+1]
        w_align = width_array[w1_index:w2_index+1]
        align_combos = product(h_align, w_align)
        grid_sectors = [label_dict[x] for x in align_combos]
        grid_sectors = set(grid_sectors)
        if grid:
            composite = self.show_grid(object_id, height_array, width_array)  
            return(composite, grid_sectors)
        return grid_sectors

    def show_grid(self, object_id, height_array, width_array):
        mygrid = Image.new(mode='1', size=(self.img_width,self.img_height))
        draw=ImageDraw.Draw(mygrid)
        for i in width_array:
            draw.line((i, 0, i, self.img_height), fill="white")
        for i in height_array:
            draw.line((0, i, self.img_width, i), fill="white")
        composite = Image.composite(mygrid, self.show_mask(self.get_mask(object_id)), mygrid)
        return composite

        
#     def grid_coords2(self, object_id, height=3, width=3):
#         """Get grid coordinates using the mask pixels in form 'A1' where 'A1' is the top left grid"""
        
    
#     def object_summary(self, object_id):
    
#     def summary(self):
        
        
    

        
        
