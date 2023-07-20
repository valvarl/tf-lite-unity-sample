using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using Cysharp.Threading.Tasks;

[RequireComponent(typeof(WebCamInput))]
public class HumanPoseSample : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string fileName = "lightweight_baseline_choi.tflite";

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField, Range(0f, 1f)]
    private float lineThickness = 0.5f;

    [SerializeField]
    private bool runBackground;

    private TensorFlowLite.HumanPose humanPose;
    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    private UniTask<bool> task;
    private TensorFlowLite.HumanPose.Result[] results;
    private CancellationToken cancellationToken;

    private void Start()
    {
        humanPose = new TensorFlowLite.HumanPose(fileName);

        draw = new PrimitiveDraw(Camera.main, gameObject.layer)
        {
            color = Color.green,
        };

        cancellationToken = this.GetCancellationTokenOnDestroy();

        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.AddListener(OnTextureUpdate);
    }

    private void OnDestroy()
    {
        var webCamInput = GetComponent<WebCamInput>();
        webCamInput.OnTextureUpdate.RemoveListener(OnTextureUpdate);

        humanPose?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        DrawResult(results);
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync(texture);
            }
        }
        else
        {
            Invoke(texture);
        }
    }

    private void DrawResult(TensorFlowLite.HumanPose.Result[] results)
    {
        if (results == null || results.Length == 0)
        {
            return;
        }

        var rect = cameraView.GetComponent<RectTransform>();
        rect.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var connections = TensorFlowLite.HumanPose.Connections;
        int len = connections.GetLength(0);
        for (int i = 0; i < len; i++)
        {
            var a = results[(int)connections[i, 0]];
            var b = results[(int)connections[i, 1]];
            draw.Line3D(
                MathTF.Lerp(min, max, new Vector3(a.x, 1f - a.y, 0)),
                MathTF.Lerp(min, max, new Vector3(b.x, 1f - b.y, 0)),
                lineThickness
            );

        }

        draw.Apply();
    }

    private void Invoke(Texture texture)
    {
        humanPose.Invoke(texture);
        results = humanPose.GetResults();
        cameraView.material = humanPose.transformMat;
    }

    private async UniTask<bool> InvokeAsync(Texture texture)
    {
        results = await humanPose.InvokeAsync(texture, cancellationToken);
        cameraView.material = humanPose.transformMat;
        return true;
    }
}