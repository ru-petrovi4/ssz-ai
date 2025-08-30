using System;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

// == Adam: класс для шага обновления параметров ==
// Используется для обучения матриц W12 и W21
public class Adam
{
    // m,v — внутренние состояния для скользящих средних градиентов
    private readonly float[] m, v;
    private readonly float beta1, beta2, eps;
    private long t; // номер шага

    public Adam(int size, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        m = new float[size];
        v = new float[size];
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        t = 0;
    }

    // Один шаг Adam для всего массива параметров
    // w — параметры, grad — соответствующие градиенты, lr — шаг
    public void Step(float[] w, float[] grad, float lr)
    {
        t++; // увеличиваем номер шага
        float b1t = 1f - (float)Math.Pow(beta1, t),
              b2t = 1f - (float)Math.Pow(beta2, t);

        for (int i = 0; i < w.Length; i++)
        {
            float gi = grad[i];
            m[i] = beta1 * m[i] + (1f - beta1) * gi;         // обновление среднего градиента
            v[i] = beta2 * v[i] + (1f - beta2) * gi * gi;    // накопление среднего квадрата

            float mhat = m[i] / b1t, vhat = v[i] / b2t;      // bias-correction
            w[i] -= lr * (mhat / ((float)Math.Sqrt(vhat) + eps)); // обновление параметра
        }
    }
}