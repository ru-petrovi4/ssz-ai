using Ssz.AI.Helpers;
using System;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

class BiDirectionalMapperAnchored
{
    public readonly int Dim;

    // Параметры якоря
    private float s;               // фиксированный масштаб вдоль e1: ||y*|| / ||x*||
    private MatrixFloat Hx;        // d x d, переводит x_hat -> e1
    private MatrixFloat Hy;        // d x d, переводит e1 -> y_hat
    private MatrixFloat W;         // (d-1) x (d-1), обучаемая часть (ортогональная проекция)

    // Модели
    public MatrixFloat A;          // F12 = Hy · diag(s, W) · Hx (перестраивается каждый шаг)
    public MatrixFloat B;          // F21 — свободная матрица d x d (обучаемая как раньше)

    public BiDirectionalMapperAnchored(int dim = 300)
    {
        Dim = dim;
        A = LinAlg.I(dim);
        B = LinAlg.I(dim);
        // Hx/Hy/W будут инициализированы в FitUnsupervisedAnchored при получении якорей
        Hx = LinAlg.I(dim);
        Hy = LinAlg.I(dim);
        W = LinAlg.I(dim - 1);
        s = 1f;
    }

    public float[] F12(ReadOnlySpan<float> v) => LinAlg.MatVec(A, v);
    public float[] F21(ReadOnlySpan<float> v) => LinAlg.MatVec(B, v);

    // Построить A из текущих (s, W, Hx, Hy)
    private void RebuildA()
    {
        var Core = LinAlg.BlockDiag(s, W);
        A = LinAlg.MatMul(LinAlg.MatMul(Hy, Core), Hx);
    }

    // Несупервизируемое обучение с жёстким якорем A x* = y*
    public void FitUnsupervisedAnchored(
        MatrixFloat R, MatrixFloat E,
        ReadOnlySpan<float> xAnchor, ReadOnlySpan<float> yAnchor,
        int epochs = 600, float lr = 1e-3f, float lambdaOrtho = 1e-3f,
        bool unitNorm = true, bool center = true,
        bool projectWEvery = true, int WProjectionPeriod = 50,
        bool projectBEvery = true, int BProjectionPeriod = 50,
        Action<int, float, float, float>? onEpoch = null)
    {
        if (R.Rows() != Dim || E.Rows() != Dim) throw new ArgumentException("Data shape must be [Dim x N]");
        if (xAnchor.Length != Dim || yAnchor.Length != Dim) throw new ArgumentException("Anchor vectors must have length Dim");

        // Предобработка данных
        if (unitNorm) { LinAlg.L2NormalizeColumns(R); LinAlg.L2NormalizeColumns(E); }
        if (center) { LinAlg.CenterColumns(R); LinAlg.CenterColumns(E); }

        // Нормы и направления якорей
        float nx = MathF.Sqrt(MathF.Max(TensorPrimitives.Dot(xAnchor, xAnchor), 1e-20f));
        float ny = MathF.Sqrt(MathF.Max(TensorPrimitives.Dot(yAnchor, yAnchor), 1e-20f));
        var xHat = new float[Dim]; var xHatSpan = xHat.AsSpan();
        var yHat = new float[Dim]; var yHatSpan = yHat.AsSpan();
        TensorPrimitives.Multiply(xAnchor, 1.0f / nx, xHatSpan);
        TensorPrimitives.Multiply(yAnchor, 1.0f / ny, yHatSpan);

        // Базисные векторы
        var e1 = new float[Dim];
        e1[0] = 1f;

        // Householder: Hx: xHat -> e1; Hy: e1 -> yHat
        Hx = LinAlg.HouseholderFromTo(xHatSpan, e1);
        Hy = LinAlg.HouseholderFromTo(e1, yHatSpan);

        // Фиксированный масштаб вдоль e1
        s = ny / nx;

        // Инициализация W и A
        W = LinAlg.I(Dim - 1);
        RebuildA(); // A = Hy diag(s, W) Hx

        int NR = R.Cols(), NE = E.Cols();
        float invNR = 1f / MathF.Max(1, NR);
        float invNE = 1f / MathF.Max(1, NE);
        var I = LinAlg.I(Dim);

        for (int ep = 1; ep <= epochs; ep++)
        {
            // --------- Forward ---------
            // RU цикл: BAR - R
            var AR = LinAlg.MatMul(A, R);
            var BAR = LinAlg.MatMul(B, AR);
            var diffR = LinAlg.Sub(BAR, R);

            // EN цикл: ABE - E
            var BE = LinAlg.MatMul(B, E);
            var ABE = LinAlg.MatMul(A, BE);
            var diffE = LinAlg.Sub(ABE, E);

            float lossR = LinAlg.FroNormSq(diffR) * invNR;
            float lossE = LinAlg.FroNormSq(diffE) * invNE;

            // --------- Градиенты по A и B (как раньше) ---------
            var Rt = LinAlg.Transpose(R);
            var Et = LinAlg.Transpose(E);
            var At = LinAlg.Transpose(A);
            var Bt = LinAlg.Transpose(B);

            // gradA1 = 2 B^T (BAR - R) R^T
            var gradA1 = LinAlg.MatMul(LinAlg.MatMul(Bt, diffR), Rt);
            LinAlg.ScaleInPlace(gradA1, 2f * invNR);

            // gradA2 = 2 (ABE - E) E^T B^T
            var gradA2 = LinAlg.MatMul(LinAlg.MatMul(diffE, Et), Bt);
            LinAlg.ScaleInPlace(gradA2, 2f * invNE);

            // gradB1 = 2 (BAR - R) R^T A^T
            var gradB1 = LinAlg.MatMul(LinAlg.MatMul(diffR, Rt), At);
            LinAlg.ScaleInPlace(gradB1, 2f * invNR);

            // gradB2 = 2 A^T (ABE - E) E^T
            var gradB2 = LinAlg.MatMul(LinAlg.MatMul(At, diffE), Et);
            LinAlg.ScaleInPlace(gradB2, 2f * invNE);

            // Ортогональная регуляризация для A и B
            var AtA = LinAlg.MatMul(At, A);
            var BtB = LinAlg.MatMul(Bt, B);
            var EoA = LinAlg.Sub(AtA, I);
            var EoB = LinAlg.Sub(BtB, I);
            float lossOrtho = lambdaOrtho * (LinAlg.FroNormSq(EoA) + LinAlg.FroNormSq(EoB));

            var gradAOrtho = LinAlg.MatMul(A, EoA);
            LinAlg.ScaleInPlace(gradAOrtho, 4f * lambdaOrtho);

            var gradBOrtho = LinAlg.MatMul(B, EoB);
            LinAlg.ScaleInPlace(gradBOrtho, 4f * lambdaOrtho);

            // Итоговые градиенты по A и B
            var gradA = LinAlg.Add(gradA1, gradA2);
            LinAlg.AddInPlace(gradA, gradAOrtho);

            var gradB = LinAlg.Add(gradB1, gradB2);
            LinAlg.AddInPlace(gradB, gradBOrtho);

            // --------- Обновление B (свободное) ---------
            var stepB = LinAlg.Scale(gradB, lr);
            LinAlg.SubInPlace(B, stepB);

            // Необязательная проекция B к ортогональным
            if (projectBEvery && (ep % BProjectionPeriod == 0 || ep == epochs))
                B = LinAlg.ProjectToNearestOrthogonal(B, iters: 12);

            // --------- Обновление A через параметры (только W), якорь сохраняется точно ---------
            // Перенос градиента в Core-базис: G_core = Hy^T * gradA * Hx^T
            var HyT = LinAlg.Transpose(Hy);
            var HxT = LinAlg.Transpose(Hx);
            var Gc = LinAlg.MatMul(LinAlg.MatMul(HyT, gradA), HxT);

            // Из G_core берём подблок (1:,1:) -> градиент по W; элементы [0,*] и [*,0] зануляем (сохраняет нулевые внеблочные связи)
            var gradW = LinAlg.SubMatrix(Gc, 1, 1, Dim - 1, Dim - 1);

            // Шаг по W
            var stepW = LinAlg.Scale(gradW, lr);
            LinAlg.SubInPlace(W, stepW);

            // Необязательная проекция W к ортогональным (сохраняет изометрию на дополнении к якорной оси)
            if (projectWEvery && (ep % WProjectionPeriod == 0 || ep == epochs))
                W = LinAlg.ProjectToNearestOrthogonal(W, iters: 12);

            // Перестроить A из (s, W, Hx, Hy)
            RebuildA();

            onEpoch?.Invoke(ep, lossR, lossE, lossOrtho);
        }
    }

    public (float ruCycleMSE, float enCycleMSE) EvaluateCycleErrors(MatrixFloat R, MatrixFloat E)
    {
        var diffR = LinAlg.Sub(LinAlg.MatMul(B, LinAlg.MatMul(A, R)), R);
        var diffE = LinAlg.Sub(LinAlg.MatMul(A, LinAlg.MatMul(B, E)), E);
        float mseR = LinAlg.FroNormSq(diffR) / MathF.Max(1, R.Cols());
        float mseE = LinAlg.FroNormSq(diffE) / MathF.Max(1, E.Cols());
        return (mseR, mseE);
    }
}