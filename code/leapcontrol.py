import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import cv2, numpy

class LeapListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']
    previousID = -1;
    drawPosition = ()
    is_drawing = False
    draw_frame = []
    rois = []


    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"


    def bind(self, frame, dx, dy):
        draw =  self.draw_frame.copy();
        M = numpy.float32([[1,0,dx],[0,1,dy]])
        draw = cv2.warpAffine(draw,M,(640,480))

        dst = cv2.add(frame[0:480, 0:640], draw[0:480, 0:640])
        dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)
        if len(self.drawPosition) == 2:
            if self.is_drawing:
                cv2.circle(dst, self.drawPosition, 10, (0,0,255), -1)
            else:
                cv2.circle(dst, self.drawPosition, 10, (0,255,0), -1)

        return dst

    def clearDrawIndicator(self):
        self.drawPosition = ()

    def update_frame(self, frame, dx, dy):
        if not len(self.draw_frame):
            self.draw_frame = frame
            self.draw_frame[0:480, 0:640] = 0
        if len(self.drawPosition) == 2:
            if self.is_drawing:
                cv2.circle(self.draw_frame, self.drawPosition, 10, (255,255,255), -1)
        
        return self.bind(frame, dx, dy)

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

       # Get hands
        for hand in frame.hands:
            if not hand.is_left:
                position = hand.fingers.frontmost.tip_position
                self.drawPosition = (2 * int(position[0]), 480 - int(position[1]))
                minDistance = 100
                for finger in hand.fingers:
                    distance =  finger.tip_position[2] - position[2]
                    self.is_drawing =  distance > 0 and distance < 25
                    if distance < minDistance and distance > 0:
                        minDistance = distance
                    if self.is_drawing:
                        break


        # Get gestures
        for gesture in frame.gestures():
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                if self.previousID == gesture.id:
                    continue
                self.previousID = gesture.id
                circle = CircleGesture(gesture)

                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
                    clockwiseness = "clockwise"
                else:
                    clockwiseness = "counterclockwise"

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state != Leap.Gesture.STATE_START:
                    previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI

                print "  Circle id: %d, %s, progress: %f, radius: %f, angle: %f degrees, %s" % (
                        gesture.id, self.state_names[gesture.state],
                        circle.progress, circle.radius, swept_angle * Leap.RAD_TO_DEG, clockwiseness)

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

def main():
    # Create a sample listener and controller
    listener = LeapListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
