using OpenCvSharp;

namespace TensorFlowLite
{
    public class Config
    {
        public static int joint_num = 21;
        public static int depth_dim = 32;
        public static (int, int) input_shape = (256, 256);
        public static (int, int) output_shape = (32, 32);

        public static Scalar pixel_mean = new Scalar(0.485, 0.456, 0.406);
        public static Scalar pixel_std = new Scalar(0.229, 0.224, 0.225);
        
        public static int[] bbox_3d_shape = new int[]{ 2000, 2000, 2000 };
    }
}