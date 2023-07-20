using System;
using System.Threading;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;
using Cysharp.Threading.Tasks;

using OpenCvSharp;
using OpenCvSharp.Tracking;


namespace TensorFlowLite
{
    public class HumanPose : BaseImagePredictor<float> 
    {
        public enum Part
        {
            Head_top = 0, //0
            Thorax, //1
            R_Shoulder, //2
            R_Elbow, //3
            R_Wrist, // 4
            L_Shoulder, // 5
            L_Elbow, // 6
            L_Wrist, //7
            R_Hip, //8
            R_Knee, //9
            R_Ankle, //10
            L_Hip, // 11
            L_Knee, // 12
            L_Ankle, // 13
            Pelvis, //14
            Spine, //15
            Head, //16
            R_Hand, //17
            L_Hand, //18
            R_Toe, //19
            L_Toe //20
        }

        public static readonly Part[,] Connections = new Part[,]
        {
            // HEAD
            { Part.Head_top, Part.Head },
            { Part.Head, Part.Thorax },
            { Part.Thorax, Part.Spine },
            { Part.Spine, Part.Pelvis },
            // BODY
            { Part.Pelvis, Part.R_Hip },
            { Part.Pelvis, Part.L_Hip },
            { Part.R_Hip, Part.R_Knee },
            { Part.R_Knee, Part.R_Ankle },
            { Part.R_Ankle, Part.R_Toe },
            { Part.L_Hip, Part.L_Knee },
            { Part.L_Knee, Part.L_Ankle },
            { Part.L_Ankle, Part.L_Toe },
            { Part.Thorax, Part.R_Shoulder },
            { Part.R_Shoulder, Part.R_Elbow },
            { Part.R_Elbow, Part.R_Wrist },
            { Part.R_Wrist, Part.R_Hand },
            { Part.Thorax, Part.L_Shoulder },
            { Part.L_Shoulder, Part.L_Elbow },
            { Part.L_Elbow, Part.L_Wrist },
            { Part.L_Wrist, Part.L_Hand }
        };

        [System.Serializable]
        public struct Result
        {
            public Part part;
            public float x;
            public float y;
            public float z;
        }

        float[] focal = new float[]{ 1500, 1500 };
        float[] princpt = null;

        float root_depth = 12500;  // obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)

        Result[] results = new Result[Config.joint_num];

        float[,] img2bb_trans = new float[2, 3];
        float[,,] inputs0 = new float[3, Config.input_shape.Item1, Config.input_shape.Item2];
        float[,,] outputs0 = null;

        // tracking
        public HOGDescriptor hog;
        public Tracker tracker;
        public bool tracking = false;
        public Rect2d? trackedRect = null;

        float[,] output_pose_2d = null;
        float[,] output_pose_3d = null;

        public HumanPose(string modelPath) : base(modelPath, Accelerator.GPU)
        {
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            outputs0 = new float[odim0[1], odim0[2], odim0[3]];

            // tracking
            hog = new HOGDescriptor();
            hog.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
            tracker = Tracker.Create(TrackerTypes.KCF);
        }

        static OpenCvSharp.Rect2d? DetectHuman(Mat frame, HOGDescriptor hog)
        {
            var gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.RGB2GRAY);

            var humans = hog.DetectMultiScale(gray, winStride: new Size(4, 4), padding: new Size(8, 8), scale: 1.05);

            if (humans.Length > 0)
            {
                var rect = humans[0];
                return new Rect2d(rect.X, rect.Y, rect.Width, rect.Height);
            }

            return null;
        }

        public void GetBbox(Mat frame)
        {
            if (!tracking)
            {
                trackedRect = DetectHuman(frame, hog);

                if (trackedRect.HasValue)
                {
                    tracker.Init(frame, trackedRect.Value);
                    tracking = true;
                }
            }
            else
            {
                Rect2d bbox = (Rect2d)trackedRect;
                bool success = tracker.Update(frame, ref bbox);
                trackedRect = bbox;

                if (!(success && trackedRect.HasValue))
                {
                    tracking = false;
                }
            }
        }

        public static Mat TextureToMat(Texture texture)
        {
            if (texture == null)
            {
                Debug.LogError("Texture is null.");
                return null;
            }

            // Convert Texture to Texture2D
            RenderTexture rt = RenderTexture.GetTemporary(texture.width, texture.height);
            Graphics.Blit(texture, rt);
            RenderTexture.active = rt;

            Texture2D texture2D = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
            texture2D.ReadPixels(new UnityEngine.Rect(0, 0, rt.width, rt.height), 0, 0);
            texture2D.Apply();

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            // Get pixel data
            Color32[] colors = texture2D.GetPixels32();

            // Convert Color32[] to byte[]
            byte[] data = new byte[colors.Length * 3];
            int dataIndex = 0;
            for (int i = 0; i < colors.Length; i++)
            {
                data[dataIndex++] = colors[i].b;
                data[dataIndex++] = colors[i].g;
                data[dataIndex++] = colors[i].r;
            }

            // Create Mat from byte[] and return
            Mat mat = new Mat(texture.height, texture.width, MatType.CV_8UC3);
            Marshal.Copy(data, 0, mat.Data, data.Length);

            // Clean up
            UnityEngine.Object.Destroy(texture2D);

            return mat;
        }

        public async UniTask<Mat> TextureToMatAsync(Texture texture, CancellationToken cancellationToken = default)
        {
            if (texture == null)
            {
                Debug.LogError("Texture is null.");
                return null;
            }

            // Convert Texture to Texture2D
            RenderTexture rt = RenderTexture.GetTemporary(texture.width, texture.height);
            Graphics.Blit(texture, rt);
            RenderTexture.active = rt;

            Texture2D texture2D = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);

            await UniTask.Run(() =>
            {
                texture2D.ReadPixels(new UnityEngine.Rect(0, 0, rt.width, rt.height), 0, 0);
                texture2D.Apply();
            }, cancellationToken: cancellationToken);

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            // Get pixel data
            Color32[] colors = texture2D.GetPixels32();

            // Convert Color32[] to byte[]
            byte[] data = new byte[colors.Length * 3];
            int dataIndex = 0;
            for (int i = 0; i < colors.Length; i++)
            {
                data[dataIndex++] = colors[i].b;
                data[dataIndex++] = colors[i].g;
                data[dataIndex++] = colors[i].r;
            }

            // Create Mat from byte[] and return
            Mat mat = new Mat(texture.height, texture.width, MatType.CV_8UC3);
            Marshal.Copy(data, 0, mat.Data, data.Length);

            // Clean up
            UnityEngine.Object.Destroy(texture2D);

            return mat;
        }

        public override void Invoke(Texture inputTex)
        {
            Mat original_img = TextureToMat(inputTex);
            Preprocess(original_img, inputTex.width, inputTex.height);
            
            interpreter.SetInputTensorData(0, inputs0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            
            var pose_3d = Postprocess();
            output_pose_2d = pose_3d.Item1;
            output_pose_3d = pose_3d.Item2;
        }

        public async UniTask<Result[]> InvokeAsync(Texture inputTex, CancellationToken cancellationToken)
        {
            Mat original_img = await TextureToMatAsync(inputTex, cancellationToken);
            await UniTask.SwitchToThreadPool();

            Preprocess(original_img, inputTex.width, inputTex.height);
            
            interpreter.SetInputTensorData(0, inputs0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            
            var pose_3d = Postprocess();
            output_pose_2d = pose_3d.Item1;
            output_pose_3d = pose_3d.Item2;

            var results = GetResults();

            await UniTask.SwitchToMainThread(cancellationToken);
            return results;
        }

        public void Preprocess(Mat original_img, int width, int height)
        {
            if (princpt == null) {
                princpt = new float[]{width / 2, height / 2};
            }

            GetBbox(original_img);

            if (trackedRect == null) {
                return;
            }
            
            Rect2d? processed_bbox = Utils.ProcessBbox(trackedRect.Value, height, width, Config.input_shape);
            var patch_image = Utils.GeneratePatchImage(original_img, processed_bbox.Value, false, 1.0f, 0.0f, false, Config.input_shape);

            Mat image = patch_image.Item1;
            Mat img2bb_trans_mat = patch_image.Item2;
            
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    img2bb_trans[i, j] = (float)img2bb_trans_mat.At<double>(i, j);
                }
            }

            image.ConvertTo(image, MatType.CV_32F, 1.0, 0);
            image = image - Config.pixel_mean;
            image = Utils.DivideByScalar(image, Config.pixel_std);
            inputs0 = Utils.MatToFloatArray(image);
        }

        public (float[,], float[,]) Postprocess()
        {
            float[,] pose_3d = Utils.SoftArgmax(outputs0, Config.joint_num, Config.depth_dim, Config.output_shape);

            // Normalize the x and y coordinates
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 0] = pose_3d[i, 0] / Config.output_shape.Item2 * Config.input_shape.Item2;
                pose_3d[i, 1] = pose_3d[i, 1] / Config.output_shape.Item1 * Config.input_shape.Item1;
            }

            // Concatenate pose_3d with ones
            float[,] pose_3d_xy1 = new float[Config.joint_num, 3];
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d_xy1[i, 0] = pose_3d[i, 0];
                pose_3d_xy1[i, 1] = pose_3d[i, 1];
                pose_3d_xy1[i, 2] = 1;
            }

            // Concatenate img2bb_trans with [0, 0, 1]
            float[,] img2bb_trans_001 = new float[3, 3];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    img2bb_trans_001[i, j] = img2bb_trans[i, j];
                }
            }
            img2bb_trans_001[2, 0] = 0;
            img2bb_trans_001[2, 1] = 0;
            img2bb_trans_001[2, 2] = 1;

            // Inverse img2bb_trans_001
            float[,] img2bb_trans_001_inv = Utils.InverseMatrix3x3(img2bb_trans_001);

            // Transpose img2bb_trans_001_inv
            for (int i = 0; i < 3; i++)
            {
                for (int j = i + 1; j < 3; j++)
                {
                    float temp = img2bb_trans_001_inv[i, j];
                    img2bb_trans_001_inv[i, j] = img2bb_trans_001_inv[j, i];
                    img2bb_trans_001_inv[j, i] = temp;
                }
            }

            // Multiply matrices
            float[,] pose_3d_transformed = Utils.MultiplyMatrix(pose_3d_xy1, img2bb_trans_001_inv);

            // Update the x and y coordinates
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 0] = pose_3d_transformed[i, 0];
                pose_3d[i, 1] = pose_3d_transformed[i, 1];
            }

            float[,] output_pose_2d = (float[,])pose_3d.Clone();

            // Calculate the absolute continuous depth
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 2] = (pose_3d[i, 2] / Config.depth_dim * 2 - 1) * (Config.bbox_3d_shape[0] / 2) + root_depth;
            }

            float[,] output_pose_3d = Utils.PixelToCam(pose_3d, focal, princpt);

            return (output_pose_2d, output_pose_3d);
        }

        public Result[] GetResults()
        {
            if (output_pose_3d != null) {
                for (int i = 0; i < Config.joint_num; i++)
                {
                    results[i].x = output_pose_3d[i, 0];
                    results[i].y = output_pose_3d[i, 1];
                    results[i].z = output_pose_3d[i, 2];
                    results[i].part = (Part)i;
                }
            }
            return results;
        }
    }
}
