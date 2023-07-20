using System.Collections.Generic;
using OpenCvSharp;

namespace TensorFlowLite
{
    public class Vis
    {
        public static List<Scalar> GenerateColors(int numberOfColors)
        {
            List<Scalar> colors = new List<Scalar>();

            for (int i = 0; i < numberOfColors; i++)
            {
                float hue = 360.0f * i / numberOfColors;
                Vec3b hsv = new Vec3b((byte)hue, 255, 255);
                Scalar color = new Scalar(hsv[0], hsv[1], hsv[2]);
                Mat hsvMat = new Mat(1, 1, MatType.CV_8UC3, color);
                Mat bgrMat = new Mat();
                Cv2.CvtColor(hsvMat, bgrMat, ColorConversionCodes.HSV2BGR);
                Scalar bgrColor = bgrMat.At<Scalar>(0, 0);
                colors.Add(bgrColor);
            }

            return colors;
        }

        public static Mat VisKeyPoints(Mat img, float[,] kps, HumanPose.Part[,] kpsLines, float kpThresh = 0.4f, double alpha = 1.0)
        {
            // Get colors
            List<Scalar> colors = GenerateColors(kpsLines.GetLength(0));

            Mat kpMask = img.Clone();

            // Draw the keypoints
            for (int l = 0; l < kpsLines.GetLength(0); l++)
            {
                int i1 = (int)kpsLines[l, 0];
                int i2 = (int)kpsLines[l, 1];

                Point p1 = new Point((int)kps[i1, 0], (int)kps[i1, 1]);
                Point p2 = new Point((int)kps[i2, 0], (int)kps[i2, 1]);
                if (kps[i1, 2] > kpThresh && kps[i2, 2] > kpThresh)
                {
                    kpMask.Line(p1, p2, colors[l], 2, LineTypes.AntiAlias);
                }
                if (kps[i1, 2] > kpThresh)
                {
                    kpMask.Circle(p1, 3, colors[l], -1, LineTypes.AntiAlias);
                }
                if (kps[i2, 2] > kpThresh)
                {
                    kpMask.Circle(p2, 3, colors[l], -1, LineTypes.AntiAlias);
                }
            }

            // Blend the keypoints.
            Mat blended = new Mat();
            Cv2.AddWeighted(img, 1.0 - alpha, kpMask, alpha, 0, dst: blended);
            return blended;
        }
    }
}