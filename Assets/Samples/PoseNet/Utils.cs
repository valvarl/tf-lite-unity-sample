using System;
using System.Collections;
using System.Collections.Generic;
using OpenCvSharp;
using UnityEngine;

using System.Linq;

namespace TensorFlowLite
{
    public static class Utils
    {
        public static float[,] PixelToCam(float[,] pose_3d, float[] focal, float[] princpt)
        {
            float[,] cam_pose_3d = new float[Config.joint_num, 3];

            for (int i = 0; i < Config.joint_num; i++)
            {
                cam_pose_3d[i, 0] = (pose_3d[i, 0] - princpt[0]) * pose_3d[i, 2] / focal[0];
                cam_pose_3d[i, 1] = (pose_3d[i, 1] - princpt[1]) * pose_3d[i, 2] / focal[1];
                cam_pose_3d[i, 2] = pose_3d[i, 2];
            }

            return cam_pose_3d;
        }
        public static float[,] SoftArgmax(float[,,] heatmaps, int joint_num, int depth_dim, (int, int) output_shape)
        {
            int[] heatmapShape = new int[] { joint_num * depth_dim, output_shape.Item1, output_shape.Item2 };
            int size = heatmapShape[0] * heatmapShape[1] * heatmapShape[2];

            // Reshape the heatmaps
            float[] heatmapsFlat = new float[size];
            Buffer.BlockCopy(heatmaps, 0, heatmapsFlat, 0, size * sizeof(float));
            float[,] heatmaps2D = new float[joint_num, size / joint_num];
            for (int i = 0; i < joint_num; i++)
            {
                for (int j = 0; j < size / Config.joint_num; j++)
                {
                    heatmaps2D[i, j] = heatmapsFlat[i * depth_dim * output_shape.Item1 * output_shape.Item2 + j];
                }
            }

            // Apply Softmax
            float[,] softmaxHeatmaps = Softmax(heatmaps2D);

            // Reshape back to the original dimensions
            float[,,] heatmapsReshaped = new float[joint_num, depth_dim, output_shape.Item1 * output_shape.Item2];
            for (int i = 0; i < joint_num; i++)
            {
                for (int j = 0; j < depth_dim; j++)
                {
                    for (int k = 0; k < output_shape.Item1 * output_shape.Item2; k++)
                    {
                        heatmapsReshaped[i, j, k] = softmaxHeatmaps[i, j * output_shape.Item1 * output_shape.Item2 + k];
                    }
                }
            }

            float[] rangeX = Enumerable.Range(1, output_shape.Item1).Select(x => (float)x).ToArray();
            float[] rangeY = Enumerable.Range(1, output_shape.Item2).Select(y => (float)y).ToArray();
            float[] rangeZ = Enumerable.Range(1, depth_dim).Select(z => (float)z).ToArray();

            float[,] coord_out = new float[joint_num, 3];
            for (int i = 0; i < joint_num; i++)
            {
                float accu_x = 0, accu_y = 0, accu_z = 0;
                for (int j = 0; j < depth_dim; j++)
                {
                    for (int x = 0; x < output_shape.Item1; x++)
                    {
                        for (int y = 0; y < output_shape.Item2; y++)
                        {
                            accu_x += heatmapsReshaped[i, j, x * output_shape.Item2 + y] * rangeX[y];
                            accu_y += heatmapsReshaped[i, j, x * output_shape.Item2 + y] * rangeY[x];
                            accu_z += heatmapsReshaped[i, j, x * output_shape.Item2 + y] * rangeZ[j];
                        }
                    }
                }

                coord_out[i, 0] = accu_x - 1;
                coord_out[i, 1] = accu_y - 1;
                coord_out[i, 2] = accu_z - 1;
            }

            return coord_out;
        }

        public static float[,] Softmax(float[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);

            float[,] output = new float[rows, cols];
            float[] maxValues = new float[rows];

            // Find the maximum value for each row
            for (int i = 0; i < rows; i++)
            {
                float maxValue = float.MinValue;
                for (int j = 0; j < cols; j++)
                {
                    if (input[i, j] > maxValue)
                    {
                        maxValue = input[i, j];
                    }
                }
                maxValues[i] = maxValue;
            }

            // Calculate the exponentials and the sum of exponentials for each row
            float[] expSums = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                float sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    float expValue = (float)Math.Exp(input[i, j] - maxValues[i]);
                    output[i, j] = expValue;
                    sum += expValue;
                }
                expSums[i] = sum;
            }

            // Normalize the exponentials to obtain the softmax values
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] /= expSums[i];
                }
            }

            return output;
        }

        public static Rect2d? ProcessBbox(Rect2d bbox, int width, int height, (int, int) input_shape)
        {
            // sanitize bboxes
            double x1 = Math.Max(0, bbox.X);
            double y1 = Math.Max(0, bbox.Y);
            double x2 = Math.Min(width - 1, x1 + Math.Max(0, bbox.Width - 1));
            double y2 = Math.Min(height - 1, y1 + Math.Max(0, bbox.Height - 1));
            if (bbox.Width * bbox.Height > 0 && x2 >= x1 && y2 >= y1)
            {
                bbox = new Rect2d(x1, y1, x2 - x1, y2 - y1);
            }
            else
            {
                return null;
            }

            // aspect ratio preserving bbox
            double c_x = bbox.X + bbox.Width / 2;
            double c_y = bbox.Y + bbox.Height / 2;
            double aspect_ratio = input_shape.Item2 / input_shape.Item1;
            if (bbox.Width > aspect_ratio * bbox.Height)
            {
                bbox.Height = bbox.Width / aspect_ratio;
            }
            else if (bbox.Width < aspect_ratio * bbox.Height)
            {
                bbox.Width = bbox.Height * aspect_ratio;
            }
            bbox.Width *= 1.25;
            bbox.Height *= 1.25f;
            bbox.X = c_x - bbox.Width / 2;
            bbox.Y = c_y - bbox.Height / 2;
            return bbox;
        }

        static float[] Rotate2D(float[] pt2D, float rotRad)
        {
            float x = pt2D[0];
            float y = pt2D[1];
            float sn = (float)Math.Sin(rotRad);
            float cs = (float)Math.Cos(rotRad);
            float xx = x * cs - y * sn;
            float yy = x * sn + y * cs;
            return new float[] { xx, yy };
        }

        public static Mat GenTransFromPatchCV(float cX, float cY, float srcWidth, float srcHeight, float dstWidth, float dstHeight, float scale, float rot, bool inv = false)
        {
            // augment size with scale
            float srcW = srcWidth * scale;
            float srcH = srcHeight * scale;
            float[] srcCenter = new float[] { cX, cY };

            // augment rotation
            float rotRad = (float)Math.PI * rot / 180;
            float[] srcDowndir = Rotate2D(new float[] { 0, srcH * 0.5f }, rotRad);
            float[] srcRightdir = Rotate2D(new float[] { srcW * 0.5f, 0 }, rotRad);

            float dstW = dstWidth;
            float dstH = dstHeight;
            float[] dstCenter = new float[] { dstW * 0.5f, dstH * 0.5f };
            float[] dstDowndir = new float[] { 0, dstH * 0.5f };
            float[] dstRightdir = new float[] { dstW * 0.5f, 0 };

            Point2f[] src = new Point2f[]
            {
                new Point2f(srcCenter[0], srcCenter[1]),
                new Point2f(srcCenter[0] + srcDowndir[0], srcCenter[1] + srcDowndir[1]),
                new Point2f(srcCenter[0] + srcRightdir[0], srcCenter[1] + srcRightdir[1])
            };

            Point2f[] dst = new Point2f[]
            {
                new Point2f(dstCenter[0], dstCenter[1]),
                new Point2f(dstCenter[0] + dstDowndir[0], dstCenter[1] + dstDowndir[1]),
                new Point2f(dstCenter[0] + dstRightdir[0], dstCenter[1] + dstRightdir[1])
            };

            Mat trans;
            if (inv)
            {
                trans = Cv2.GetAffineTransform(dst, src);
            }
            else
            {
                trans = Cv2.GetAffineTransform(src, dst);
            }

            return trans;
        }

        public static (Mat, Mat) GeneratePatchImage(Mat cvimg, Rect2d bbox, bool doFlip, float scale, float rot, bool doOcclusion, (int, int) input_shape)
        {
            Mat img = cvimg.Clone();
            int imgHeight = img.Rows, imgWidth = img.Cols;

            // synthetic occlusion
            if (doOcclusion)
            {
                while (true)
                {
                    float areaMin = 0.0f;
                    float areaMax = 0.7f;
                    float synthArea = (float)(new System.Random().NextDouble() * (areaMax - areaMin) + areaMin) * (float)(bbox.Width * bbox.Height);

                    float ratioMin = 0.3f;
                    float ratioMax = 1 / 0.3f;
                    float synthRatio = (float)(new System.Random().NextDouble() * (ratioMax - ratioMin) + ratioMin);

                    float synthH = (float)Math.Sqrt(synthArea * synthRatio);
                    float synthW = (float)Math.Sqrt(synthArea / synthRatio);
                    float synthXmin = (float)(new System.Random().NextDouble() * ((float)bbox.Width - synthW - 1) + (float)bbox.X);
                    float synthYmin = (float)(new System.Random().NextDouble() * ((float)bbox.Height - synthH - 1) + (float)bbox.Y);

                    if (synthXmin >= 0 && synthYmin >= 0 && synthXmin + synthW < imgWidth && synthYmin + synthH < imgHeight)
                    {
                        int xmin = (int)synthXmin;
                        int ymin = (int)synthYmin;
                        int w = (int)synthW;
                        int h = (int)synthH;
                        Mat occlusion = new Mat(h, w, MatType.CV_32FC3, new OpenCvSharp.Scalar(255, 255, 255)) * new Mat(h, w, MatType.CV_32FC3, new System.Random().NextDouble());
                        occlusion.CopyTo(img.SubMat(ymin, ymin + h, xmin, xmin + w));
                        break;
                    }
                }
            }

            float bb_c_x = (float)(bbox.X + 0.5 * bbox.Width);
            float bb_c_y = (float)(bbox.Y + 0.5 * bbox.Height);
            float bb_width = (float)bbox.Width;
            float bb_height = (float)bbox.Height;

            if (doFlip)
            {
                img = img.Flip(FlipMode.Y);
                bb_c_x = imgWidth - bb_c_x - 1;
            }

            Mat trans = GenTransFromPatchCV(bb_c_x, bb_c_y, bb_width, bb_height, input_shape.Item2, input_shape.Item1, scale, rot, inv: false);
            Mat imgPatch = new Mat();
            Cv2.WarpAffine(img, imgPatch, trans, new Size(input_shape.Item2, input_shape.Item1), flags: InterpolationFlags.Linear);

            return (imgPatch, trans);
        }

        public static Texture2D ToTexture2D(Texture rTex)
        {
            Texture2D dest = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
            dest.Apply(false);
            Graphics.CopyTexture(rTex, dest);
            return dest;
        }

        public static Mat DivideByScalar(Mat srcImage, Scalar weights)
        {
            // Создание матрицы весов
            Mat weightMatrix = new Mat(srcImage.Rows, srcImage.Cols, srcImage.Type(), weights);

            // Выполнение поэлементного деления
            Mat resultImage = new Mat();
            Cv2.Divide(srcImage, weightMatrix, resultImage);

            return resultImage;
        }

        public static float[,,] MatToFloatArray(Mat mat)
        {
            if (mat.Empty())
            {
                throw new ArgumentException("The input Mat object is empty.");
            }

            if (mat.Type() != MatType.CV_32FC1 && mat.Type() != MatType.CV_32FC3 && mat.Type() != MatType.CV_32FC4)
            {
                mat.ConvertTo(mat, MatType.CV_32F);
            }

            int rows = mat.Rows;
            int cols = mat.Cols;
            int channels = mat.Channels();
            float[,,] floatArray = new float[channels, rows, cols];

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    float[] pixel = new float[channels];
                    mat.GetArray(row, col, pixel);
                    for (int ch = 0; ch < channels; ch++)
                    {
                        floatArray[ch, row, col] = pixel[ch];
                    }
                }
            }

            return floatArray;
        }

                public static float[,] InverseMatrix3x3(float[,] matrix)
        {
            float determinant = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[2, 1] * matrix[1, 2]) -
                                matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) +
                                matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]);

            float invDet = 1 / determinant;

            float[,] invMatrix = new float[3, 3];
            invMatrix[0, 0] = (matrix[1, 1] * matrix[2, 2] - matrix[2, 1] * matrix[1, 2]) * invDet;
            invMatrix[0, 1] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) * invDet;
            invMatrix[0, 2] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) * invDet;
            invMatrix[1, 0] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) * invDet;
            invMatrix[1, 1] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) * invDet;
            invMatrix[1, 2] = (matrix[1, 0] * matrix[0, 2] - matrix[0, 0] * matrix[1, 2]) * invDet;
            invMatrix[2, 0] = (matrix[1, 0] * matrix[2, 1] - matrix[2, 0] * matrix[1, 1]) * invDet;
            invMatrix[2, 1] = (matrix[2, 0] * matrix[0, 1] - matrix[0, 0] * matrix[2, 1]) * invDet;
            invMatrix[2, 2] = (matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]) * invDet;

            return invMatrix;
        }

        public static float[,] MultiplyMatrix(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
            {
                throw new InvalidOperationException("Matrix dimensions are not valid for multiplication.");
            }

            float[,] result = new float[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < colsA; k++)
                    {
                        sum += A[i, k] * B[k, j];
                    }
                    result[i, j] = sum;
                }
            }

            return result;
        }
    }
}
