using Ssz.AI.Models;
using System;
using System.Numerics.Tensors;

namespace Ssz.AI.Helpers;

static class LinAlg
{
    public static int Rows(this MatrixFloat A) => A.Dimensions[0];
    public static int Cols(this MatrixFloat A) => A.Dimensions[1];

    public static MatrixFloat I(int n)
    {
        var M = new MatrixFloat(n, n);
        for (int i = 0; i < n; i++) M[i, i] = 1f;
        return M;
    }

    public static MatrixFloat Zeros(int r, int c) => new MatrixFloat(r, c);

    public static MatrixFloat Transpose(MatrixFloat A)
    {
        int r = A.Rows(), c = A.Cols();
        var AT = new MatrixFloat(c, r);
        for (int j = 0; j < c; j++)
        {
            var src = A.GetColumn(j);
            for (int i = 0; i < r; i++)
                AT[j, i] = src[i];
        }
        return AT;
    }

    // C = A * B (column-major; C[:,j] = sum_p A[:,p] * B[p,j])
    public static MatrixFloat MatMul(MatrixFloat A, MatrixFloat B)
    {
        int r = A.Rows(), k = A.Cols(), kc = B.Rows(), c = B.Cols();
        if (k != kc) throw new ArgumentException("MatMul: shape mismatch");

        var C = new MatrixFloat(r, c);
        var tmp = new float[r];
        var tmpSpan = tmp.AsSpan();

        for (int p = 0; p < k; p++)
        {
            var aCol = A.GetColumn(p);
            for (int j = 0; j < c; j++)
            {
                float s = B[p, j];
                if (s == 0f) continue;
                TensorPrimitives.Multiply(aCol, s, tmpSpan);
                var cCol = C.GetColumn(j);
                TensorPrimitives.Add(cCol, tmpSpan, cCol);
            }
        }
        return C;
    }

    public static MatrixFloat Add(MatrixFloat A, MatrixFloat B)
    {
        if (A.Data.Length != B.Data.Length) throw new ArgumentException("Add: shape mismatch");
        var C = new MatrixFloat(A.Dimensions);
        TensorPrimitives.Add(A.Data, B.Data, C.Data);
        return C;
    }

    public static MatrixFloat Sub(MatrixFloat A, MatrixFloat B)
    {
        if (A.Data.Length != B.Data.Length) throw new ArgumentException("Sub: shape mismatch");
        var C = new MatrixFloat(A.Dimensions);
        TensorPrimitives.Subtract(A.Data, B.Data, C.Data);
        return C;
    }

    public static MatrixFloat Scale(MatrixFloat A, float s)
    {
        var C = new MatrixFloat(A.Dimensions);
        TensorPrimitives.Multiply(A.Data, s, C.Data);
        return C;
    }

    public static void AddInPlace(MatrixFloat A, MatrixFloat B)
    {
        if (A.Data.Length != B.Data.Length) throw new ArgumentException("AddInPlace: shape mismatch");
        TensorPrimitives.Add(A.Data, B.Data, A.Data);
    }

    public static void SubInPlace(MatrixFloat A, MatrixFloat B)
    {
        if (A.Data.Length != B.Data.Length) throw new ArgumentException("SubInPlace: shape mismatch");
        TensorPrimitives.Subtract(A.Data, B.Data, A.Data);
    }

    public static void ScaleInPlace(MatrixFloat A, float s)
    {
        TensorPrimitives.Multiply(A.Data, s, A.Data);
    }

    public static float FroNormSq(MatrixFloat A) => TensorPrimitives.Dot(A.Data, A.Data);

    public static void L2NormalizeColumns(MatrixFloat M, float eps = 1e-8f)
    {
        int r = M.Rows(), c = M.Cols();
        for (int j = 0; j < c; j++)
        {
            var col = M.GetColumn(j);
            float ss = TensorPrimitives.Dot(col, col);
            float nrm = MathF.Sqrt(MathF.Max(ss, eps));
            TensorPrimitives.Multiply(col, 1.0f / nrm, col);
        }
    }

    public static void CenterColumns(MatrixFloat M)
    {
        int r = M.Rows(), c = M.Cols();
        var mean = new float[r];
        var meanSpan = mean.AsSpan();
        meanSpan.Clear();
        for (int j = 0; j < c; j++)
            TensorPrimitives.Add(meanSpan, M.GetColumn(j), meanSpan);
        TensorPrimitives.Multiply(meanSpan, 1.0f / MathF.Max(1, c), meanSpan);
        for (int j = 0; j < c; j++)
            TensorPrimitives.Subtract(M.GetColumn(j), meanSpan, M.GetColumn(j));
    }

    // Householder H(a→b): H a = b, где u = (a - b)/||a - b||, H = I - 2 u u^T
    public static MatrixFloat HouseholderFromTo(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int d = a.Length;
        if (b.Length != d) throw new ArgumentException("Householder: len mismatch");
        var u = new float[d];
        var uSpan = u.AsSpan();

        // u = a - b
        TensorPrimitives.Subtract(a, b, uSpan);
        float n2 = TensorPrimitives.Dot(uSpan, uSpan);
        if (n2 <= 1e-20f)
        {
            // a == b -> H = I
            return I(d);
        }
        float invN = 1.0f / MathF.Sqrt(n2);
        TensorPrimitives.Multiply(uSpan, invN, uSpan);

        var H = I(d);
        var tmp = new float[d];
        var tmpSpan = tmp.AsSpan();

        // H = I - 2 u u^T: по столбцам
        for (int j = 0; j < d; j++)
        {
            var col = H.GetColumn(j);
            TensorPrimitives.Multiply(uSpan, u[j], tmpSpan);   // tmp = u * u_j
            TensorPrimitives.Multiply(tmpSpan, 2f, tmpSpan);   // 2*u*u_j
            TensorPrimitives.Subtract(col, tmpSpan, col);      // col -= 2*u*u_j
        }
        return H;
    }

    // Блочная диагональ Core = diag(s, W), result d x d
    public static MatrixFloat BlockDiag(float s, MatrixFloat W)
    {
        int d1 = 1, d2 = W.Rows();
        if (W.Rows() != W.Cols()) throw new ArgumentException("W must be square");
        int d = d1 + d2;
        var M = new MatrixFloat(d, d);
        M[0, 0] = s;
        for (int j = 0; j < d2; j++)
            for (int i = 0; i < d2; i++)
                M[i + 1, j + 1] = W[i, j];
        return M;
    }

    // Вырезка подматрицы (r x c) от (i0, j0)
    public static MatrixFloat SubMatrix(MatrixFloat A, int i0, int j0, int r, int c)
    {
        var S = new MatrixFloat(r, c);
        for (int j = 0; j < c; j++)
            for (int i = 0; i < r; i++)
                S[i, j] = A[i0 + i, j0 + j];
        return S;
    }

    // Вклейка подматрицы B в A по смещению (i0, j0)
    public static void SetSubMatrix(MatrixFloat A, int i0, int j0, MatrixFloat B)
    {
        for (int j = 0; j < B.Cols(); j++)
            for (int i = 0; i < B.Rows(); i++)
                A[i0 + i, j0 + j] = B[i, j];
    }

    // Применение матрицы к вектору: y = A v
    public static float[] MatVec(MatrixFloat A, ReadOnlySpan<float> v)
    {
        int r = A.Rows(), c = A.Cols();
        if (c != v.Length) throw new ArgumentException("MatVec: shape mismatch");
        var y = new float[r];
        var ySpan = y.AsSpan();
        ySpan.Clear();

        var tmp = new float[r];
        var tmpSpan = tmp.AsSpan();
        for (int j = 0; j < c; j++)
        {
            float s = v[j];
            if (s == 0f) continue;
            var aCol = A.GetColumn(j);
            TensorPrimitives.Multiply(aCol, s, tmpSpan);
            TensorPrimitives.Add(ySpan, tmpSpan, ySpan);
        }
        return y;
    }

    // Полярная проекция (Ньютон–Шульц) к ближайшей ортогональной: X_{k+1} = 0.5 X (3I - X^T X)
    public static MatrixFloat ProjectToNearestOrthogonal(MatrixFloat M, int iters = 12)
    {
        if (M.Rows() != M.Cols()) throw new ArgumentException("Project: square only");
        int d = M.Rows();
        float fn = MathF.Sqrt(MathF.Max(FroNormSq(M), 1e-12f));
        var X = Scale(M, 1f / fn);
        var Iden = I(d);

        for (int t = 0; t < iters; t++)
        {
            var Xt = Transpose(X);
            var XtX = MatMul(Xt, X);
            var threeI = Scale(Iden, 3f);
            SubInPlace(threeI, XtX);
            var Xn = MatMul(X, threeI);
            ScaleInPlace(Xn, 0.5f);
            X = Xn;
        }
        return X;
    }
}