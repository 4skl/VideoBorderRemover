#TODO : test -g and -color option
#! code under development
import numpy as np
import cv2, sys, time
output_file='output.avi' #more compatible
codec='XVID' #more compatible
mode=0#black borders
color=None
if len(sys.argv) < 2:
    print(sys.argv[0],'<input filename> [-o <output filename>] [-c <output codec name ex : XVID, MJPG, X264,... (can need libs, and special output filename, like .mp4,..)>] [<-g grayscale_value>|<-color [r,v,b]>]')
    exit(0)

try:
    i_o = sys.argv.index('-o')
    output_file = sys.argv[i_o+1]
except ValueError:
    pass

try:
    i_c = sys.argv.index('-c')
    codec = sys.argv[i_c+1]
except ValueError:
    pass

try:
    i_gry = sys.argv.index('-g')
    mode=1
    color = sys.argv[i_gry+1]
except ValueError:
    pass

def str_to_intlist(s):
    return list(map(int, s.strip('][').split(', ')))

try:
    i_col = sys.argv.index('-color')
    mode=3
    color = sys.argv[i_col+1]
except ValueError:
    pass




'''frame :
[[v v v v v],
[v v v v v]
....]
frame[-1][0] is vertically one upper
frame[0][-1] is vertically one lefter
'''
#add line verif
#correct len call time loss ?
threshold = 5#really small (tf is r + v + b (r,v,b in [0;255]))


def point_near_color(point, max_threshold, threshold_function=lambda l : np.sum(l)):
    #fastest : return np.any(point)
    return threshold_function(point)<=max_threshold

#detect first index in an vertical line scanning the frame horizontally with a given height where the threshold is higher than max_thresoshold
#actual threshold function more usefull with a max_threshold multiple of brush_height
#add placing
#for black boder
def search_scan_threashold(frame, max_threshold, threshold_function=lambda l: np.sum(l), brush_height=1, vertical=False, jump=1, inversed=False): #threshold_function should accept a list of any size (based on height)
    #writen in horizontal scanning style
    fwidth = len(frame[0])
    fheight = len(frame)
    if vertical:
        fwidth, fheight = fheight, fwidth
    row_i = 0
    row_max = fheight-brush_height
    while row_i <= row_max:
        for col_i in (range(fwidth-1,-1,-1) if inversed else range(fwidth)):
            if vertical:
                if threshold_function(frame[col_i, row_i:row_i+brush_height]) >= max_threshold:
                    return col_i
            else:
                if threshold_function(frame[row_i:row_i+brush_height, col_i]) >= max_threshold:
                    return col_i
        row_i += jump

#for grayscale border
def search_scan_minmaxthreashold(frame, max_threshold, min_threshold, threshold_function=lambda l: np.sum(l), brush_height=1, vertical=False, jump=1, inversed=False): #threshold_function should accept a list of any size (based on height)
    #writen in horizontal scanning style
    fwidth = len(frame[0])
    fheight = len(frame)
    if vertical:
        fwidth, fheight = fheight, fwidth
    row_i = 0
    row_max = fheight-brush_height
    while row_i <= row_max:
        for col_i in (range(fwidth-1,-1,-1) if inversed else range(fwidth)):
            if vertical:
                if not (min_threshold < threshold_function(frame[col_i, row_i:row_i+brush_height]) < max_threshold):
                    return col_i
            else:
                if not (min_threshold < threshold_function(frame[row_i:row_i+brush_height, col_i]) < max_threshold):
                    return col_i
        row_i += jump

def colorAvg(l):
    v_len = len(l[0])
    v_count = len(l)
    avg = [0]*v_len
    for v in l:
        for i in range(v_len):
            avg[i] += v[i]
    for i in range(v_len):
        avg[i] /= v_count
    return avg

def sqErrorColorFunction(c1, c2):
    err = 0
    for i in range(len(c1)):
        err += (c1[i]-c2[i])**2
    return err

def search_scan_nearvalthreashold(frame, val, threshold, brush_height=1, vertical=False, jump=1, inversed=False): #threshold_function should accept a list of any size (based on height)
    sqThreshold = threshold**2
    #writen in horizontal scanning style
    fwidth = len(frame[0])
    fheight = len(frame)
    if vertical:
        fwidth, fheight = fheight, fwidth
    row_i = 0
    row_max = fheight-brush_height
    while row_i <= row_max:
        for col_i in (range(fwidth-1,-1,-1) if inversed else range(fwidth)):
            if vertical:
                if sqErrorColorFunction(colorAvg(frame[col_i, row_i:row_i+brush_height]), val) >= sqThreshold:
                    return col_i
            else:
                if sqErrorColorFunction(colorAvg(frame[row_i:row_i+brush_height, col_i]), val)  >= sqThreshold:
                    return col_i
        row_i += jump

def get_max_underth_outbox(frame, threshold=6, brush_size=5, jump_multiplier=1):#faster, only max threshold
    max_threshold = brush_size*threshold
    #jump=brush_size way faster
    jump=brush_size*jump_multiplier
    left = search_scan_threashold(frame, max_threshold, brush_height=brush_size, jump=jump) 
    right = search_scan_threashold(frame, max_threshold, brush_height=brush_size, inversed=True, jump=jump)
    top = search_scan_threashold(frame, max_threshold, brush_height=brush_size, vertical=True, jump=jump)
    bottom = search_scan_threashold(frame, max_threshold, brush_height=brush_size, vertical=True, inversed=True, jump=jump)
    return (top, left, bottom, right)

def get_max_insideth_outbox(frame, max_threshold=6, min_threshold=6, brush_size=5, jump_multiplier=1):#slower, not only max threshold, min threshold added
    max_threshold = brush_size*threshold
    min_threshold = brush_size*threshold
    #jump=brush_size way faster
    jump=brush_size*jump_multiplier
    left = search_scan_minmaxthreashold(frame, max_threshold, min_threshold, brush_height=brush_size, jump=jump) 
    right = search_scan_minmaxthreashold(frame, max_threshold, min_threshold, brush_height=brush_size, inversed=True, jump=jump)
    top = search_scan_minmaxthreashold(frame, max_threshold, min_threshold, brush_height=brush_size, vertical=True, jump=jump)
    bottom = search_scan_minmaxthreashold(frame, max_threshold, min_threshold, brush_height=brush_size, vertical=True, inversed=True, jump=jump)
    return (top, left, bottom, right)

def get_max_nearvalth_outbox(frame, val, threshold=6, brush_size=5, jump_multiplier=1):#slower, not only max threshold, min threshold added
    max_threshold = brush_size*threshold
    min_threshold = brush_size*threshold
    #jump=brush_size way faster
    jump=brush_size*jump_multiplier
    left = search_scan_nearvalthreashold(frame, val, threshold, brush_height=brush_size, jump=jump) 
    right = search_scan_nearvalthreashold(frame, val, threshold, brush_height=brush_size, inversed=True, jump=jump)
    top = search_scan_nearvalthreashold(frame, val, threshold, brush_height=brush_size, vertical=True, jump=jump)
    bottom = search_scan_nearvalthreashold(frame, val, threshold, brush_height=brush_size, vertical=True, inversed=True, jump=jump)
    return (top, left, bottom, right)

#dim = (top, left, bot, right)
def crop_frame(frame, valid_dim):
    return frame[valid_dim[0]:valid_dim[2], valid_dim[1]:valid_dim[3]]

def is_valid_dim(dim, min_area_size):
    if None in dim or dim[2]-dim[0] < min_area_size or dim[3]-dim[1] < min_area_size :
        return False
    return True

def remove_border(frame, threshold=6, min_area_size=256, brush_size=16):
    #check if not already removed
    '''if point_near_color(frame[0,0], max_threshold=threshold) and point_near_color(frame[len(frame)-1,len(frame[0])-1], max_threshold=threshold):
        return frame
    el'''
    #brush_size * jump_multiplier should be lower than minimal width of searched square
    jump_multiplier = int(min_area_size/brush_size)
    valid_dim = get_max_underth_outbox(frame, threshold=threshold, brush_size=brush_size, jump_multiplier=jump_multiplier)
    if not is_valid_dim(valid_dim, min_area_size):
        return frame
    return crop_frame(frame, valid_dim)

def remove_grayscale_border(frame, grayscale_value, threshold=6, min_area_size=256, brush_size=16):
    #brush_size * jump_multiplier should be lower than minimal width of searched square
    jump_multiplier = int(min_area_size/brush_size)
    valid_dim = get_max_insideth_outbox(frame, max_threshold=grayscale_value+threshold, min_threshold=grayscale_value-threshold, brush_size=brush_size, jump_multiplier=jump_multiplier)
    if not is_valid_dim(valid_dim, min_area_size):
        return frame
    return crop_frame(frame, valid_dim)

def remove_colored_border(frame, color, threshold=6, min_area_size=256, brush_size=16):
    #brush_size * jump_multiplier should be lower than minimal width of searched square
    jump_multiplier = int(min_area_size/brush_size)
    valid_dim = get_max_nearvalth_outbox(frame, color, threshold=threshold, brush_size=brush_size, jump_multiplier=jump_multiplier)
    if not is_valid_dim(valid_dim, min_area_size):
        return frame
    return crop_frame(frame, valid_dim)


cap = cv2.VideoCapture(sys.argv[1])
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width, height = 1280, 720 #! for perfs

base_fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(output_file,fourcc, base_fps , (width,height))
i = 0
l_i = 0
st_time = time.perf_counter()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        try:
            new_frame = None
            if mode == 0:
                new_frame = remove_border(frame)
            elif mode == 1:
                new_frame = remove_grayscale_border(frame, color)
            elif mode == 2:
                new_frame = get_max_nearvalth_outbox(frame, color)
            out.write(cv2.resize(new_frame,(width,height)))
        except Exception as e:
            out.write(frame)
            print(e)
            print('Error at frame ', i, f'( {round(i/base_fps, 2)}s )')
        
        i+=1
        t_a = time.perf_counter()
        if t_a-st_time>1:
            fps = (i-l_i)/(t_a-st_time)
            l_i = i
            st_time = t_a
            print(f'{i/frame_count*100:.2f}% : {i} frames done ( {round(i/base_fps, 2)}s )')
            print('speed : ', round(fps,2), f'fps ( {round(fps/base_fps, 2)}x original speed)')
    else:
        break
cap.release()
out.release()


''' oldcode
def detect_topleft(frame, max_threshold):
    if point_near_color(frame[0][0], max_threshold):
        return (0,0)
    for d in range(len(frame)):
        if point_near_color(frame[d][d], max_threshold):
            break
    #frame[d][d] is the fist nonzero in diag
    if point_near_color(frame[d-1][d], max_threshold):#the frame above is non zero implies that we hit the the left border
        left = d
        for top in range(d-1, 0, -1):
            if not point_near_color(frame[top][left], max_threshold): # we have now our top
                break
        return (top+1, left)
    elif point_near_color(frame[d][d-1], max_threshold):#the frame at left is non zero implies that we hit the the top border
        #print('top hit')
        top = d
        for left in range(d-1, 0, -1):
            if not point_near_color(frame[top][left], max_threshold): # we have now our top
                break
        return (top, left+1)

def detect_bot(frame, topleft, max_threshold):
    mh = len(frame)-1
    for bot in range(mh, topleft[0], -1):
        if point_near_color(frame[bot][topleft[1]], max_threshold):
            break
    return bot

def detect_right(frame, topleft, max_threshold):
    mw = len(frame[0])-1
    for right in range(mw, topleft[1], -1):
        if point_near_color(frame[topleft[0]][right], max_threshold):
            break
    return right

def detect_content_box(frame, max_threshold):
    topleft = detect_topleft(frame, max_threshold)
    botright = (None, None)
    if None in topleft:
        return (None, None)
        botright = (detect_bot(frame, topleft, max_threshold), detect_right(frame, topleft, max_threshold)) #detect_botright(frame)
    return (topleft[0], topleft[1], botright[0], botright[1])

def remove_border_s(frame, threshold=6, min_area_size=100, brush_size=70):
    #check if not already removed
#    if point_near_color(frame[0,0], max_threshold=threshold) and point_near_color(frame[len(frame)-1,len(frame[0])-1], max_threshold=threshold):
#        return frame
#    el
    #brush_size * jump_multiplier should be lower than minimal width of searched square
    valid_dim = detect_content_box(frame, threshold)
    if not is_valid_dim(valid_dim, min_area_size):
        jump_multiplier = 1#min_area_size/brush_size 
        valid_dim = get_max_underth_outbox(frame, threshold=threshold, brush_size=brush_size, jump_multiplier=jump_multiplier)
    if not is_valid_dim(valid_dim, min_area_size):
        return frame
    return crop_frame(frame, valid_dim)
'''