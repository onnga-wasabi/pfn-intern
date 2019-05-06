import copy

import cv2

import util
from finger import Finger, setup_fingers
from settings import FINGER_TYPES, WINDOW_SIZE

# Mode
VIEW = 'View'
EDIT = 'Edit'
EDIT_ROOT = 'Edit Root'
SEQUENTIAL = 'Sequential'


def draw_edge(finger):
    [cv2.drawMarker(img, (x, y), finger.color, 0, 10) for (x, y) in finger.key_points]
    [cv2.line(img, (x1, y1), (x2, y2), finger.color, 1, 0) for ((x1, y1), (x2, y2)) in finger.edges]


def update_window(fingers, active_finger=''):
    global img

    img = base_img.copy()
    cv2.putText(img, f'Mode: {mode} {active_finger}', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 0, 0), thickness=1)
    [draw_edge(finger) for finger in fingers]


def set_key_points(event, x, y, flags, param):
    global mode, active, temp_finger

    if event == cv2.EVENT_LBUTTONUP:
        if mode == EDIT:
            cv2.drawMarker(img, (x, y), temp_finger.color, 0, 10)
            temp_finger.add_point((x, y))
            draw_edge(temp_finger)

            if len(temp_finger) == 4:
                mode = VIEW
                fingers[active] = temp_finger
                update_window(fingers)

        elif mode == EDIT_ROOT:
            mode = VIEW
            [finger.key_points.insert(0, (x, y)) for finger in fingers]
            update_window(fingers)

        elif mode == SEQUENTIAL:
            if active == -1:
                [finger.key_points.insert(0, (x, y)) for finger in fingers]
                update_window(fingers)
                active += 1
                temp_finger = Finger(color=fingers[active].color).add_point(fingers[active].key_points[0])
                update_window([finger for i, finger in enumerate(fingers) if i != active], FINGER_TYPES[active])
            else:
                cv2.drawMarker(img, (x, y), temp_finger.color, 0, 10)
                temp_finger.add_point((x, y))
                draw_edge(temp_finger)

                if len(temp_finger) == 4:
                    fingers[active] = temp_finger
                    update_window(fingers)
                    active += 1
                    if active == 5:
                        mode = VIEW
                        update_window(fingers)
                    else:
                        temp_finger = Finger(color=fingers[active].color).add_point(fingers[active].key_points[0])
                        update_window([finger for i, finger in enumerate(fingers) if i != active], FINGER_TYPES[active])


def setup_window():
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', set_key_points)
    cv2.resizeWindow('image', WINDOW_SIZE, WINDOW_SIZE)


if __name__ == '__main__':

    imgs = [(cv2.resize(cv2.imread(str(fname)), (224, 224)), fname) for fname in util.get_files()]
    if not imgs:
        print('No data requires annotation')
        import sys
        sys.exit()

    base_img, fname = imgs.pop()
    fingers = setup_fingers()

    mode = VIEW
    setup_window()
    update_window(fingers)
    count = 0

    while(True):
        cv2.imshow('image', img)
        k = cv2.waitKey(10) & 0xFF  # 64bit

        if mode == VIEW:  # Default Mode
            if k == ord(''):  # ESC
                cv2.putText(img, f'Exit? (y)es', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 0, 0), thickness=1)
                cv2.imshow('image', img)
                k = cv2.waitKey(0) & 0xFF  # 64bit
                if k == ord('y'):
                    break
                else:
                    update_window(fingers)

            elif k == ord('n'):
                if not imgs:
                    break
                mode = VIEW
                base_img, fname = imgs.pop()
                count += 1
                print(count)
                fingers = setup_fingers()
                update_window(fingers)

            elif k in range(ord('0'), ord('5')):
                mode = EDIT
                active = int(chr(k))
                temp_finger = Finger(color=fingers[active].color).add_point(fingers[active].key_points[0])
                update_window([finger for i, finger in enumerate(fingers) if i != active], FINGER_TYPES[active])

            elif k == ord('r'):
                mode = EDIT_ROOT
                current_root = [finger.key_points.pop(0) for finger in fingers]
                update_window(fingers)

            elif k == ord('a'):
                mode = SEQUENTIAL
                old_fingers = copy.deepcopy(fingers)
                [finger.key_points.pop(0) for finger in fingers]
                update_window(fingers, 'Root')
                active = -1

            elif k == ord('s'):
                util.save_coordinates(fingers, fname.name)
                cv2.putText(img, f'Saved coordinates', (5, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 0, 0), thickness=1)

        elif mode == EDIT:
            if k == ord(''):  # ESC
                mode = VIEW
                update_window(fingers)

        elif mode == EDIT_ROOT:
            if k == ord(''):  # ESC
                mode = VIEW
                [finger.key_points.insert(0, r) for finger, r in zip(fingers, current_root)]
                update_window(fingers)

        elif mode == SEQUENTIAL:
            if k == ord(''):  # ESC
                mode = VIEW
                fingers = old_fingers
                update_window(fingers)

    cv2.destroyAllWindows()
