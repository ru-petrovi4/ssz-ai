using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils.Logging;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public sealed class BilingualMapper
{
    private ILoggersSet _loggersSet;
    private readonly int d;
    public MatrixFloat W12 { get; private set; }
    public MatrixFloat W21 { get; private set; }

    // == Применить преобразование F12 к одному вектору
    public void ApplyF12(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W12, src, dst);

    // == Применить преобразование F21 к одному вектору
    public void ApplyF21(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W21, src, dst);

    public BilingualMapper(int dim, ILoggersSet loggersSet)
    {
        _loggersSet = loggersSet;
        d = dim;
        W12 = LinAlg.Identity(d);
        W21 = LinAlg.Identity(d);
    }

    //public static void NormalizeColumns(MatrixFloat M)
    //{
    //    int cols = M.Dimensions[1];
    //    for (int j = 0; j < cols; j++)
    //    {
    //        var col = M.GetColumn(j);
    //        LinAlg.NormalizeColumnInPlace(col);
    //    }
    //}

    // core helper: add gradient for L = 1 - cos(W x, target)
    private void AddCosineLossGrad(MatrixFloat W, ReadOnlySpan<float> x, ReadOnlySpan<float> target, MatrixFloat grad, ref float loss)
    {
        var u = ArrayPool<float>.Shared.Rent(d);
        var gu = ArrayPool<float>.Shared.Rent(d);
        try
        {
            var uSpan = u.AsSpan(0, d);
            var guSpan = gu.AsSpan(0, d);

            LinAlg.MatVec(W, x, uSpan);
            float n = LinAlg.Norm(uSpan) + 1e-8f;
            float s = TensorPrimitives.Dot(uSpan, target);
            float cos = s / n;
            loss += (1f - cos);

            // grad wrt u: - ( v/n - (s/n^3) u )
            // gu = (-1/n)*v + (s/n^3)*u
            LinAlg.Copy(target, guSpan);
            TensorPrimitives.Divide(guSpan, -n, guSpan);     // (-1/n)*v
            float beta = s / (n * n * n);
            LinAlg.Saxpy(guSpan, uSpan, beta);               // + beta*u

            // grad wrt W: gu ⊗ x^T
            LinAlg.OuterAddInPlace(grad, guSpan, x, 1f);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(u);
            ArrayPool<float>.Shared.Return(gu);
        }
    }

    // cycle loss for RU: r -> a = W12 r -> u2 = W21 a, target r
    private void AddCycleRuGrad(ReadOnlySpan<float> r, MatrixFloat grad12, MatrixFloat grad21, ref float loss, float weight)
    {
        var a = ArrayPool<float>.Shared.Rent(d);
        var u2 = ArrayPool<float>.Shared.Rent(d);
        var g2 = ArrayPool<float>.Shared.Rent(d);
        var ga = ArrayPool<float>.Shared.Rent(d);
        try
        {
            var aS = a.AsSpan(0, d);
            var u2S = u2.AsSpan(0, d);
            var g2S = g2.AsSpan(0, d);
            var gaS = ga.AsSpan(0, d);

            LinAlg.MatVec(W12, r, aS);
            LinAlg.MatVec(W21, aS, u2S);

            float n = LinAlg.Norm(u2S) + 1e-8f;
            float s = TensorPrimitives.Dot(u2S, r);
            float cos = s / n;
            loss += weight * (1f - cos);

            // g2 = - ( r/n - (s/n^3) u2 )
            LinAlg.Copy(r, g2S);
            TensorPrimitives.Divide(g2S, -n, g2S);
            float beta = s / (n * n * n);
            LinAlg.Saxpy(g2S, u2S, beta);

            // grad W21 += g2 ⊗ a^T
            LinAlg.OuterAddInPlace(grad21, g2S, aS, weight);

            // ga = W21^T * g2
            // compute gaS = (W21^T) * g2 -> equivalently ga[i] = dot(W21[:,i], g2)
            for (int i = 0; i < d; i++)
            {
                var col = W21.GetColumn(i);
                gaS[i] = TensorPrimitives.Dot(col, g2S);
            }

            // grad W12 += ga ⊗ r^T
            LinAlg.OuterAddInPlace(grad12, gaS, r, weight);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(u2);
            ArrayPool<float>.Shared.Return(g2);
            ArrayPool<float>.Shared.Return(ga);
        }
    }

    // cycle loss for EN: e -> a = W21 e -> u2 = W12 a, target e
    private void AddCycleEnGrad(ReadOnlySpan<float> e, MatrixFloat grad12, MatrixFloat grad21, ref float loss, float weight)
    {
        var a = ArrayPool<float>.Shared.Rent(d);
        var u2 = ArrayPool<float>.Shared.Rent(d);
        var g2 = ArrayPool<float>.Shared.Rent(d);
        var ga = ArrayPool<float>.Shared.Rent(d);
        try
        {
            var aS = a.AsSpan(0, d);
            var u2S = u2.AsSpan(0, d);
            var g2S = g2.AsSpan(0, d);
            var gaS = ga.AsSpan(0, d);

            LinAlg.MatVec(W21, e, aS);
            LinAlg.MatVec(W12, aS, u2S);

            float n = LinAlg.Norm(u2S) + 1e-8f;
            float s = TensorPrimitives.Dot(u2S, e);
            float cos = s / n;
            loss += weight * (1f - cos);

            // g2 = - ( e/n - (s/n^3) u2 )
            LinAlg.Copy(e, g2S);
            TensorPrimitives.Divide(g2S, -n, g2S);
            float beta = s / (n * n * n);
            LinAlg.Saxpy(g2S, u2S, beta);

            // grad W12 += g2 ⊗ a^T
            LinAlg.OuterAddInPlace(grad12, g2S, aS, weight);

            // ga = W12^T * g2
            for (int i = 0; i < d; i++)
            {
                var col = W12.GetColumn(i);
                gaS[i] = TensorPrimitives.Dot(col, g2S);
            }

            // grad W21 += ga ⊗ e^T
            LinAlg.OuterAddInPlace(grad21, gaS, e, weight);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(u2);
            ArrayPool<float>.Shared.Return(g2);
            ArrayPool<float>.Shared.Return(ga);
        }
    }

    private static void AddTieGrad(MatrixFloat W12, MatrixFloat W21, MatrixFloat g12, MatrixFloat g21, float lambda)
    {
        int d = W12.Dimensions[0];
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                float diff = W21[i, j] - W12[j, i]; // (W21 - W12^T)[i,j]
                g21[i, j] += 2f * lambda * diff;
                g12[j, i] -= 2f * lambda * diff;   // equivalent to 2*(W12 - W21^T)
            }
        }
    }

    private static void AddOrthoRegGrad(MatrixFloat W, MatrixFloat g, float lambda)
    {
        // grad += 4 * W * (W^T W - I)
        int d = W.Dimensions[0];
        var G = LinAlg.Gram(W);
        for (int i = 0; i < d; i++) G[i, i] -= 1f;

        var WG = LinAlg.MatMul(W, G);
        for (int j = 0; j < d; j++)
        {
            var gj = g.GetColumn(j);
            var wgj = WG.GetColumn(j);
            for (int i = 0; i < d; i++)
                gj[i] += 4f * lambda * wgj[i];
        }
    }

    public void Train(
        MatrixFloat ru, MatrixFloat en,
        int[] seedRuIdx, int[] seedEnIdx,
        int epochs = 10,
        int batchSup = 1024,
        int batchMono = 2048,
        float lr = 0.05f,
        float wSup = 1.0f,
        float wCycle = 1.0f,
        float wTie = 0.1f,
        float wOrtho = 0.0f,
        int projectEvery = 1,
        int orthoIters = 2,
        int seed = 42)
    {
        if (seedRuIdx.Length != seedEnIdx.Length) throw new ArgumentException("Seeds length mismatch");
        var rnd = new Random(seed);
        int nRu = ru.Dimensions[1];
        int nEn = en.Dimensions[1];

        var g12 = new MatrixFloat(new[] { d, d });
        var g21 = new MatrixFloat(new[] { d, d });

        var x = new float[d];
        var y = new float[d];

        Span<float> x_ = stackalloc float[d];
        Span<float> y_ = stackalloc float[d];
        Span<float> z_ = stackalloc float[d];

        for (int ep = 0; ep < epochs; ep++)
        {
            g12.Clear();
            g21.Clear();
            float loss = 0f;

            // supervised batch
            int supCount = Math.Min(batchSup, seedRuIdx.Length);
            for (int t = 0; t < supCount; t++)
            {
                int k = rnd.Next(seedRuIdx.Length);
                var r = ru.GetColumn(seedRuIdx[k]);
                var e = en.GetColumn(seedEnIdx[k]);

                // L_sup_ru2en: 1 - cos(W12 r, e)
                AddCosineLossGrad(W12, r, e, g12, ref loss);
                // L_sup_en2ru: 1 - cos(W21 e, r)
                AddCosineLossGrad(W21, e, r, g21, ref loss);
            }
            loss *= wSup;

            // monolingual cycle RU
            int monoRu = Math.Min(batchMono, nRu);
            for (int t = 0; t < monoRu; t++)
            {
                int idx = rnd.Next(nRu);
                var r = ru.GetColumn(idx);
                AddCycleRuGrad(r, g12, g21, ref loss, wCycle);
            }

            // monolingual cycle EN
            int monoEn = Math.Min(batchMono, nEn);
            for (int t = 0; t < monoEn; t++)
            {
                int idx = rnd.Next(nEn);
                var e = en.GetColumn(idx);
                AddCycleEnGrad(e, g12, g21, ref loss, wCycle);
            }

            // tie regularization: W21 ~ W12^T
            if (wTie > 0f)
                AddTieGrad(W12, W21, g12, g21, wTie);

            // optional orthogonal regularization
            if (wOrtho > 0f)
            {
                AddOrthoRegGrad(W12, g12, wOrtho);
                AddOrthoRegGrad(W21, g21, wOrtho);
            }

            // SGD update
            for (int j = 0; j < d; j++)
            {
                var w12j = W12.GetColumn(j);
                var w21j = W21.GetColumn(j);
                var g12j = g12.GetColumn(j);
                var g21j = g21.GetColumn(j);

                for (int i = 0; i < d; i++)
                {
                    w12j[i] -= lr * g12j[i];
                    w21j[i] -= lr * g21j[i];
                }
            }

            // project to orthogonal (keeps maps near-inverses)
            if ((ep + 1) % projectEvery == 0)
            {
                LinAlg.OrthogonalizeInPlaceNewtonSchulz(W12, orthoIters);
                LinAlg.OrthogonalizeInPlaceNewtonSchulz(W21, orthoIters);
            }



            float errorRu = 1.0f;
            for (int j = 0; j < ru.Dimensions[1]; j++)
            {
                var col = ru.GetColumn(j);
                LinAlg.Copy(col, x_);

                LinAlg.MatVec(W12, x_, y_);   // y = W12 x
                LinAlg.MatVec(W21, y_, z_);   // z = W21 y -- результат обратного преобразования
                var e = TensorPrimitives.CosineSimilarity(z_, x_); // ошибка восстановления                
                if (e < errorRu)
                    errorRu = e;
            }

            float errorEn = 1.0f;
            for (int j = 0; j < en.Dimensions[1]; j++)
            {
                var col = en.GetColumn(j);
                LinAlg.Copy(col, x_);

                LinAlg.MatVec(W21, x_, y_);   // y = W12 x
                LinAlg.MatVec(W12, y_, z_);   // z = W21 y -- результат обратного преобразования                
                var e = TensorPrimitives.CosineSimilarity(z_, x_);
                if (e < errorEn)
                    errorEn = e;
            }

            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch {ep} done. Worst Cosine Ru: {errorRu}; Worst Cosine En: {errorEn}");
        }
    }

    // Find nearest EN index for a RU vector via cosine
    public int PredictRuToEnIndex(MatrixFloat en, ReadOnlySpan<float> ruVec)
    {
        var y = new float[d];
        LinAlg.MatVec(W12, ruVec, y);
        float n = LinAlg.Norm(y) + 1e-8f;
        int best = -1;
        float bestSim = float.NegativeInfinity;
        for (int j = 0; j < en.Dimensions[1]; j++)
        {
            var e = en.GetColumn(j);
            float sim = TensorPrimitives.Dot(y, e) / n; // cosine since e is normalized
            if (sim > bestSim) { bestSim = sim; best = j; }
        }
        return best;
    }

    // Find nearest RU index for an EN vector via cosine
    public int PredictEnToRuIndex(MatrixFloat ru, ReadOnlySpan<float> enVec)
    {
        var y = new float[d];
        LinAlg.MatVec(W21, enVec, y);
        float n = LinAlg.Norm(y) + 1e-8f;
        int best = -1;
        float bestSim = float.NegativeInfinity;
        for (int j = 0; j < ru.Dimensions[1]; j++)
        {
            var r = ru.GetColumn(j);
            float sim = TensorPrimitives.Dot(y, r) / n;
            if (sim > bestSim) { bestSim = sim; best = j; }
        }
        return best;
    }
}

//public class NNAligner
//{
//    // Пара генераторов: F12: dxd, F21: dxd
//    public class Generators
//    {
//        public MatrixFloat W12; // d x d
//        public MatrixFloat b12; // d x 1
//        public MatrixFloat W21; // d x d
//        public MatrixFloat b21; // d x 1
//        public Generators(int d)
//        {
//            W12 = new MatrixFloat(d, d);
//            b12 = new MatrixFloat(d, 1);
//            W21 = new MatrixFloat(d, d);
//            b21 = new MatrixFloat(d, 1);
//        }
//    }

//    // Линейные дискриминаторы: s = w^T z + b; p = sigmoid(s)
//    public class Discriminators
//    {
//        public float[] W_en; public float b_en; // на EN-пространстве
//        public float[] W_ru; public float b_ru; // на RU-пространстве
//        public Discriminators(int d)
//        {
//            W_en = new float[d];
//            W_ru = new float[d];
//        }
//    }

//    // Параметры обучения
//    public class TrainConfig
//    {
//        public int Dim = 300;
//        public int Epochs = 200;
//        public int BatchSize = 256;
//        public float LrGen = 1e-3f;
//        public float LrDisc = 1e-3f;
//        public float LambdaCycle = 10f;    // вес циклической потери
//        public float LambdaAdv = 1f;       // вес адверсариальной потери генераторов
//        public int Seed = 42;
//        public int DegreeOfParallelism = 0; // 0 => по умолчанию
//    }

//    // == Применить преобразование F12 к одному вектору
//    public void ApplyF12(ReadOnlySpan<float> src, Span<float> dst)
//        => LinAlg.MatVec(G.W12, src, dst);

//    // == Применить преобразование F21 к одному вектору
//    public void ApplyF21(ReadOnlySpan<float> src, Span<float> dst)
//        => LinAlg.MatVec(G.W21, src, dst);

//    public Generators G;

//    private ILoggersSet _loggersSet;

//    public NNAligner(ILoggersSet loggersSet)
//    {
//        this._loggersSet = loggersSet;
//        G = new Generators(300);
//    }

//    // Главный метод: обучение нейросетей F12/F21
//    public (Generators G, Discriminators D) Train(
//        MatrixFloat ru, MatrixFloat en, TrainConfig cfg)
//    {
//        if (ru.Dimensions[0] != cfg.Dim || en.Dimensions[0] != cfg.Dim)
//            throw new ArgumentException("Ожидаются матрицы Dim x N с колонками-эмбеддингами");

//        // Нормализация входных столбцов => корректная косинусная метрика
//        //NormalizeColumns(ru, cfg.DegreeOfParallelism);
//        //NormalizeColumns(en, cfg.DegreeOfParallelism);

//        int d = cfg.Dim;
//        var rnd = new Random(cfg.Seed);
//        G = InitGenerators(d, rnd);   // случайная линейная инициализация
//        var D = new Discriminators(d);    // нулевые дискриминаторы

//        // Adam моменты
//        var optG = new Adam(G, cfg);
//        var optD = new AdamDisc(D, cfg);

//        int nRU = ru.Dimensions[1], nEN = en.Dimensions[1];
//        int steps = Math.Max(1, Math.Max(nRU, nEN) / cfg.BatchSize); // ИСПРАВЛЕНО: Гарантируем >= 1 шаг

//        var po = new ParallelOptions();
//        if (cfg.DegreeOfParallelism > 0) po.MaxDegreeOfParallelism = cfg.DegreeOfParallelism;

//        var ruIdx = new int[nRU]; for (int i = 0; i < nRU; i++) ruIdx[i] = i;
//        var enIdx = new int[nEN]; for (int i = 0; i < nEN; i++) enIdx[i] = i;


//        Span<float> y = stackalloc float[d];
//        Span<float> xrec = stackalloc float[d];
//        Span<float> dL_xrec = stackalloc float[d];
//        Span<float> dL_y = stackalloc float[d];
//        Span<float> x = stackalloc float[d];
//        Span<float> yrec = stackalloc float[d];
//        Span<float> dL_yrec = stackalloc float[d];
//        Span<float> dL_x = stackalloc float[d];

//        Span<float> x_ = stackalloc float[d];
//        Span<float> y_ = stackalloc float[d];
//        Span<float> z_ = stackalloc float[d];

//        var ru_to_en = new MatrixFloat(d, cfg.BatchSize); // F12(ru)
//        var en_to_ru = new MatrixFloat(d, cfg.BatchSize); // F21(en)

//        var gradW_en = new float[d]; float gradb_en = 0f;
//        var gradW_ru = new float[d]; float gradb_ru = 0f;


//        for (int epoch = 0; epoch < cfg.Epochs; epoch++)
//        {
//            Shuffle(ruIdx, rnd);
//            Shuffle(enIdx, rnd);

//            for (int step = 0; step < steps; step++)
//            {
//                // 1) Сэмплы батча
//                var batchRU = SampleBatch(ru, ruIdx, step, cfg.BatchSize);
//                var batchEN = SampleBatch(en, enIdx, step, cfg.BatchSize);

//                // 2) Прямой проход генераторов

//                ApplyLinear(G.W12, G.b12, batchRU, ru_to_en, po);
//                ApplyLinear(G.W21, G.b21, batchEN, en_to_ru, po);

//                // 3) Обновление дискриминаторов (линейная логрегрессия)
//                Array.Clear(gradW_en); gradb_en = 0f;
//                Array.Clear(gradW_ru); gradb_ru = 0f;

//                // EN дискриминатор
//                AccumulateDiscGrad(batchEN, 1f, D.W_en, D.b_en, gradW_en, ref gradb_en);
//                AccumulateDiscGrad(ru_to_en, 0f, D.W_en, D.b_en, gradW_en, ref gradb_en);

//                // RU дискриминатор
//                AccumulateDiscGrad(batchRU, 1f, D.W_ru, D.b_ru, gradW_ru, ref gradb_ru);
//                AccumulateDiscGrad(en_to_ru, 0f, D.W_ru, D.b_ru, gradW_ru, ref gradb_ru);

//                // Шаг оптимизации дискриминаторов
//                optD.Step(gradW_en, ref gradb_en, gradW_ru, ref gradb_ru);

//                // 4) Генератор: циклическая потеря + адверсариальная
//                var gW12 = new MatrixFloat(d, d); // градиенты генераторов
//                var gb12 = new float[d];
//                var gW21 = new MatrixFloat(d, d);
//                var gb21 = new float[d];

//                // 4a) RU цикл: x -> y -> x_rec
//                for (int j = 0; j < batchRU.Dimensions[1]; j++)
//                {
//                    x = batchRU.GetColumn(j);

//                    // ИСПРАВЛЕНО: Используем 'd' вместо 300

//                    Lin(W: G.W12, b: G.b12, x, y);


//                    Lin(W: G.W21, b: G.b21, y, xrec);


//                    CosineOneMinusGradWrtSecond(x, xrec, cfg.LambdaCycle, dL_xrec);

//                    OuterAdd(dL_xrec, y, gW21);
//                    TensorPrimitives.Add(gb21, dL_xrec, gb21);


//                    MatVecT(G.W21, dL_xrec, dL_y);

//                    float s = TensorPrimitives.Dot(D.W_en, y) + D.b_en;
//                    float p = Sigmoid(s);
//                    float coeff = cfg.LambdaAdv * (p - 1f);
//                    FmaInPlace(D.W_en, coeff, dL_y);

//                    OuterAdd(dL_y, x, gW12);
//                    TensorPrimitives.Add(gb12, dL_y, gb12);
//                }

//                // 4b) EN цикл: y0 -> x -> y_rec
//                for (int j = 0; j < batchEN.Dimensions[1]; j++)
//                {
//                    var y0 = batchEN.GetColumn(j);

//                    // ИСПРАВЛЕНО: Используем 'd' вместо 300

//                    Lin(G.W21, G.b21, y0, x);


//                    Lin(G.W12, G.b12, x, yrec);


//                    CosineOneMinusGradWrtSecond(y0, yrec, cfg.LambdaCycle, dL_yrec);

//                    OuterAdd(dL_yrec, x, gW12);
//                    TensorPrimitives.Add(gb12, dL_yrec, gb12);


//                    MatVecT(G.W12, dL_yrec, dL_x);

//                    float s2 = TensorPrimitives.Dot(D.W_ru, x) + D.b_ru;
//                    float p2 = Sigmoid(s2);
//                    float coeff2 = cfg.LambdaAdv * (p2 - 1f);
//                    FmaInPlace(D.W_ru, coeff2, dL_x);

//                    OuterAdd(dL_x, y0, gW21);
//                    TensorPrimitives.Add(gb21, dL_x, gb21);
//                }

//                // 5) Шаг оптимизации генераторов
//                optG.Step(gW12, gb12, gW21, gb21);
//            }

//            float errorRu = 1.0f;
//            for (int j = 0; j < ru.Dimensions[1]; j++)
//            {
//                var col = ru.GetColumn(j);
//                LinAlg.Copy(col, x_);

//                LinAlg.MatVec(G.W12, x_, y_);   // y = W12 x
//                LinAlg.MatVec(G.W21, y_, z_);   // z = W21 y -- результат обратного преобразования
//                var e = TensorPrimitives.CosineSimilarity(z_, x_); // ошибка восстановления                
//                if (e < errorRu)
//                    errorRu = e;
//            }

//            float errorEn = 1.0f;
//            for (int j = 0; j < en.Dimensions[1]; j++)
//            {
//                var col = en.GetColumn(j);
//                LinAlg.Copy(col, x_);

//                LinAlg.MatVec(G.W21, x_, y_);   // y = W12 x
//                LinAlg.MatVec(G.W12, y_, z_);   // z = W21 y -- результат обратного преобразования                
//                var e = TensorPrimitives.CosineSimilarity(z_, x_);
//                if (e < errorEn)
//                    errorEn = e;
//            }

//            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch {epoch} done. Worst Cosine Ru: {errorRu}; Worst Cosine En: {errorEn}");

//            //var r1 = new float[300];
//            //var r2 = new float[300];
//            //var e1 = new float[300];
//            //var e2 = new float[300];
//            //for (int i = 50; i < 55; i++)
//            //{
//            //    var ruW = ru.GetColumn(i);
//            //    ApplyF12(ruW, r1);
//            //    ApplyF21(r1, r2);
//            //    var dot = TensorPrimitives.CosineSimilarity(ruW, r2);
//            //    var enIndex = LinAlg.NearestColumnIndex(en, r1);
//            //    if (enIndex < LanguageInfo_EN.Words.Count)
//            //        _loggersSet.UserFriendlyLogger.LogInformation($"RU: F21(F12(v)) cosine: {dot}; RU: {LanguageInfo_RU.Words[i].Name}; EN: {LanguageInfo_EN.Words[enIndex].Name}");
//            //    else
//            //        _loggersSet.UserFriendlyLogger.LogInformation($"RU: F21(F12(v)) cosine: {dot}; EN: ---");

//            //    var enW = en.GetColumn(i);
//            //    ApplyF21(enW, e1);
//            //    ApplyF12(e1, e2);
//            //    dot = TensorPrimitives.CosineSimilarity(enW, e2);
//            //    int ruIndex = LinAlg.NearestColumnIndex(ruEmb, e1);
//            //    if (ruIndex < LanguageInfo_RU.Words.Count)
//            //        _loggersSet.UserFriendlyLogger.LogInformation($"EN: F12(F21(v)) cosine: {dot}; EN: {LanguageInfo_EN.Words[i].Name}; RU: {LanguageInfo_RU.Words[ruIndex].Name}");
//            //    else
//            //        _loggersSet.UserFriendlyLogger.LogInformation($"EN: F12(F21(v)) cosine: {dot}; EN: ---");
//            //}
//        }

//        return (G, D);
//    }

//    // ------------------------------------------------------------
//    // Линал и векторные примитивы с TensorPrimitives
//    // ------------------------------------------------------------

//    private void Lin(MatrixFloat W, MatrixFloat b, ReadOnlySpan<float> x, Span<float> y)
//    {
//        b.GetColumn(0).CopyTo(y);
//        for (int c = 0; c < W.Dimensions[1]; c++)
//        {
//            float scale = x[c];
//            if (scale != 0f)
//                TensorPrimitives.FusedMultiplyAdd(W.GetColumn(c), scale, y, y);
//        }
//    }

//    private void ApplyLinear(MatrixFloat W, MatrixFloat b, MatrixFloat X, MatrixFloat Y, ParallelOptions po)
//    {
//        Parallel.For(0, X.Dimensions[1], po, j =>
//        {
//            Lin(W, b, X.GetColumn(j), Y.GetColumn(j));
//        });
//    }

//    private void FmaInPlace(ReadOnlySpan<float> v, float alpha, Span<float> y)
//    {
//        TensorPrimitives.FusedMultiplyAdd(v, alpha, y, y);
//    }

//    private void OuterAdd(ReadOnlySpan<float> u, ReadOnlySpan<float> v, MatrixFloat dW)
//    {
//        for (int c = 0; c < dW.Dimensions[1]; c++)
//        {
//            float scale = v[c];
//            if (scale != 0f)
//                TensorPrimitives.FusedMultiplyAdd(u, scale, dW.GetColumn(c), dW.GetColumn(c));
//        }
//    }

//    private void MatVecT(MatrixFloat W, ReadOnlySpan<float> g, Span<float> z)
//    {
//        for (int c = 0; c < W.Dimensions[1]; c++)
//            z[c] = TensorPrimitives.Dot(W.GetColumn(c), g);
//    }

//    private void CosineOneMinusGradWrtSecond(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float scale, Span<float> grad)
//    {
//        float na = TensorPrimitives.Norm(a);
//        float nb = TensorPrimitives.Norm(b);
//        if (na <= 1e-9f || nb <= 1e-9f) { grad.Clear(); return; }

//        float cos = TensorPrimitives.Dot(a, b); // VALFIX
//        float C1 = scale * cos / (nb * nb);
//        float C2 = -scale / (na * nb);

//        // grad = C1 * b + C2 * a
//        TensorPrimitives.Multiply(a, C2, grad); // grad = C2 * a
//        TensorPrimitives.FusedMultiplyAdd(b, C1, grad, grad); // grad += C1 * b
//    }

//    private void AccumulateDiscGrad(MatrixFloat batch, float t, ReadOnlySpan<float> W, float b, float[] gradW, ref float gradb)
//    {
//        for (int j = 0; j < batch.Dimensions[1]; j++)
//        {
//            var z = batch.GetColumn(j);
//            float delta = Sigmoid(TensorPrimitives.Dot(W, z) + b) - t;
//            TensorPrimitives.FusedMultiplyAdd(z, delta, gradW, gradW);
//            gradb += delta;
//        }
//    }

//    private float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

//    private void NormalizeColumns(MatrixFloat X, int dop = 0)
//    {
//        var po = new ParallelOptions();
//        if (dop > 0) po.MaxDegreeOfParallelism = dop;
//        Parallel.For(0, X.Dimensions[1], po, j =>
//        {
//            var col = X.GetColumn(j);
//            float norm = TensorPrimitives.Norm(col);
//            if (norm > 1e-9f)
//                TensorPrimitives.Multiply(col, 1f / norm, col);
//        });
//    }

//    private Generators InitGenerators(int d, Random rnd)
//    {
//        var G = new Generators(d);
//        for (int j = 0; j < d; j++)
//        {
//            var c12 = G.W12.GetColumn(j);
//            var c21 = G.W21.GetColumn(j);
//            for (int i = 0; i < d; i++)
//            {
//                c12[i] = (float)(rnd.NextDouble() * 0.02 - 0.01);
//                c21[i] = (float)(rnd.NextDouble() * 0.02 - 0.01);
//            }
//        }
//        return G;
//    }

//    private void Shuffle(int[] a, Random rnd)
//    {
//        for (int i = a.Length - 1; i > 0; i--)
//        {
//            int j = rnd.Next(i + 1);
//            (a[i], a[j]) = (a[j], a[i]);
//        }
//    }

//    private MatrixFloat SampleBatch(MatrixFloat X, int[] order, int step, int batchSize)
//    {
//        int d = X.Dimensions[0];
//        int n = X.Dimensions[1];
//        if (n == 0) return new MatrixFloat(d, 0);

//        var B = new MatrixFloat(d, batchSize);
//        int start = (step * batchSize) % n;
//        for (int k = 0; k < batchSize; k++)
//        {
//            int idx = order[(start + k) % n];
//            X.GetColumn(idx).CopyTo(B.GetColumn(k));
//        }
//        return B;
//    }

//    #region Adam Optimizers
//    private class Adam
//    {
//        private readonly float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
//        private readonly float lr;
//        private readonly MatrixFloat mW12, vW12, mW21, vW21;
//        private readonly float[] mb12, vb12, mb21, vb21;
//        private int t = 0;

//        private Generators G;
//        public Adam(Generators G, TrainConfig cfg)
//        {
//            this.G = G;
//            lr = cfg.LrGen;
//            mW12 = new MatrixFloat(G.W12.Dimensions); vW12 = new MatrixFloat(G.W12.Dimensions);
//            mW21 = new MatrixFloat(G.W21.Dimensions); vW21 = new MatrixFloat(G.W21.Dimensions);
//            mb12 = new float[G.b12.Dimensions[0]]; vb12 = new float[G.b12.Dimensions[0]];
//            mb21 = new float[G.b21.Dimensions[0]]; vb21 = new float[G.b21.Dimensions[0]];
//        }

//        public void Step(MatrixFloat gW12, float[] gb12, MatrixFloat gW21, float[] gb21)
//        {
//            t++;
//            AdamUpdate(G.W12.Data, mW12.Data, vW12.Data, gW12.Data);
//            AdamUpdate(G.W21.Data, mW21.Data, vW21.Data, gW21.Data);
//            AdamUpdate(G.b12.GetColumn(0), mb12, vb12, gb12);
//            AdamUpdate(G.b21.GetColumn(0), mb21, vb21, gb21);
//        }

//        private void AdamUpdate(Span<float> w, Span<float> m, Span<float> v, ReadOnlySpan<float> g)
//        {
//            float m_corr = 1f - MathF.Pow(beta1, t);
//            float v_corr = 1f - MathF.Pow(beta2, t);
//            for (int i = 0; i < w.Length; i++)
//            {
//                float gi = g[i];
//                m[i] = beta1 * m[i] + (1 - beta1) * gi;
//                v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
//                float mhat = m[i] / m_corr;
//                float vhat = v[i] / v_corr;
//                w[i] -= lr * (mhat / (MathF.Sqrt(vhat) + eps));
//            }
//        }
//        private void AdamUpdate(Span<float> w, float[] m, float[] v, ReadOnlySpan<float> g) => AdamUpdate(w, m.AsSpan(), v.AsSpan(), g);
//    }

//    private class AdamDisc
//    {
//        private readonly float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, lr;
//        private readonly float[] mW_en, vW_en, mW_ru, vW_ru;
//        private float mb_en, vb_en, mb_ru, vb_ru;
//        private int t = 0;
//        private Discriminators D;

//        public AdamDisc(Discriminators D, TrainConfig cfg)
//        {
//            this.D = D;
//            lr = cfg.LrDisc;
//            mW_en = new float[D.W_en.Length]; vW_en = new float[D.W_en.Length];
//            mW_ru = new float[D.W_ru.Length]; vW_ru = new float[D.W_ru.Length];
//        }

//        public void Step(float[] gW_en, ref float gb_en, float[] gW_ru, ref float gb_ru)
//        {
//            t++;
//            AdamUpdate(D.W_en, mW_en, vW_en, gW_en);
//            AdamUpdate(D.W_ru, mW_ru, vW_ru, gW_ru);
//            (D.b_en, mb_en, vb_en) = AdamUpdateScalar(D.b_en, mb_en, vb_en, gb_en);
//            (D.b_ru, mb_ru, vb_ru) = AdamUpdateScalar(D.b_ru, mb_ru, vb_ru, gb_ru);
//        }

//        private void AdamUpdate(float[] w, float[] m, float[] v, float[] g)
//        {
//            float m_corr = 1f - MathF.Pow(beta1, t);
//            float v_corr = 1f - MathF.Pow(beta2, t);
//            for (int i = 0; i < w.Length; i++)
//            {
//                m[i] = beta1 * m[i] + (1 - beta1) * g[i];
//                v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
//                float mhat = m[i] / m_corr;
//                float vhat = v[i] / v_corr;
//                w[i] -= lr * (mhat / (MathF.Sqrt(vhat) + eps));
//            }
//            Array.Clear(g, 0, g.Length);
//        }

//        private (float w, float m, float v) AdamUpdateScalar(float w, float m, float v, float g)
//        {
//            float m_corr = 1f - MathF.Pow(beta1, t);
//            float v_corr = 1f - MathF.Pow(beta2, t);
//            m = beta1 * m + (1 - beta1) * g;
//            v = beta2 * v + (1 - beta2) * g * g;
//            float mhat = m / m_corr;
//            float vhat = v / v_corr;
//            w -= lr * (mhat / (MathF.Sqrt(vhat) + eps));
//            return (w, m, v);
//        }
//    }
//    #endregion
//}