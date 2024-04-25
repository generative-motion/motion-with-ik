**Utilizing IK for Generative Motion**

We used the transformer framework provided in "Motion In-Betweening with Two Stage Transformers" (link below) and trained it with our custom representation of motion data. Rather than representing motion as rotations of all the bones in the body, we used select control points, including the hands/feet, base of spine, top of spine and head to fully constrain the model.

With our representation, the model can more directly learn the locations of end effectors (hands and feet), leading to faster and easier training, inference, as well as more precise movements of those end effectors. Inverse kinematics is used to solve for the rest of the bones.

Although the original model used two transformer stages, ours currently uses only one and performs reasonably well with minimal foot sliding. We expect significant room for improvement by refining our armature design, loss functions and adding in the second stage.

We are also planning on implementing this tool as a plugin into Blender such that animators can have a much easier time quickly making character animations. The AI generated results can be easily modified and adjusted by the animator.

The code for the original model can be found here: [https://github.com/victorqin/motion_i...](https://github.com/victorqin/motion_inbetweening)

Our video demonstration: https://www.youtube.com/watch?v=ZrFl5hJJQ5o 
