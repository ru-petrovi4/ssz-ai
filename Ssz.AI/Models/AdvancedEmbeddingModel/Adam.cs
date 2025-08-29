using System;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class Adam
{
    private readonly float[] m;
    private readonly float[] v;
    private readonly float beta1;
    private readonly float beta2;
    private readonly float eps;
    private long t;
    
    public Adam(int size, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        m = new float[size];
        v = new float[size];
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        t = 0;
    }

    public void Step(float[] w, float[] grad, float lr)
    {
        t++;
        float b1t = 1f - MathF.Pow(beta1, t);
        float b2t = 1f - MathF.Pow(beta2, t);

        for (int i = 0; i < w.Length; i++)
        {
            float gi = grad[i];
            m[i] = beta1 * m[i] + (1f - beta1) * gi;
            v[i] = beta2 * v[i] + (1f - beta2) * gi * gi;

            float mhat = m[i] / b1t;
            float vhat = v[i] / b2t;

            w[i] -= lr * (mhat / ((float)Math.Sqrt(vhat) + eps));
        }
    }
}