using Ssz.AI.Models;
using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Helpers;

public static class LinAlg
{
    public static int Rows(this MatrixFloat A) => A.Dimensions[0];
    public static int Cols(this MatrixFloat A) => A.Dimensions[1];

    public static void Copy(ReadOnlySpan<float> src, Span<float> dst)
    {
        src.CopyTo(dst);
    }
   
    public static void Fill(Span<float> dst, float value)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] = value;
    }

    public static void AddInPlace(Span<float> dst, ReadOnlySpan<float> add)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] += add[i];
    }

    public static void SubInPlace(Span<float> dst, ReadOnlySpan<float> sub)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] -= sub[i];
    }

    public static void ScaleInPlace(Span<float> dst, float s)
    {
        TensorPrimitives.Multiply(dst, s, dst);
    }

    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        return TensorPrimitives.Dot(a, b);
    }

    // y = W * x
    public static void MatVec(MatrixFloat A, ReadOnlySpan<float> v, Span<float> y)
    {
        int r = A.Rows(), c = A.Cols();
        if (c != v.Length) throw new ArgumentException("MatVec: shape mismatch");

        y.Clear();
        for (int j = 0; j < c; j++)
        {
            float s = v[j];
            if (s == 0f) continue;
            var aCol = A.GetColumn(j);
            TensorPrimitives.MultiplyAdd(aCol, s, y, y);
        }
    }

    // y = W^T * x
    public static void MatTVec(MatrixFloat W, ReadOnlySpan<float> x, Span<float> y)
    {
        int k = W.Dimensions[1];
        for (int j = 0; j < k; j++)
        {
            var col = W.GetColumn(j);
            y[j] = TensorPrimitives.Dot(col, x);
        }
    }

    // C += alpha * A * B, для квадратных 300x300 (простая реализация)
    public static void MatMulAdd(MatrixFloat A, MatrixFloat B, MatrixFloat C, float alpha = 1f)
    {
        int d = A.Dimensions[0];
        int k = A.Dimensions[1]; // = B.Dimensions
        int n = B.Dimensions[1];
        //if (!A.Dimensions.SequenceEqual(C.Dimensions) || !B.Dimensions.SequenceEqual(C.Dimensions) || !A.Dimensions.SequenceEqual(B.Dimensions))
        //    throw new ArgumentException("Shape mismatch in MatMulAdd.");

        for (int j = 0; j < n; j++)
        {
            var cCol = C.GetColumn(j);
            for (int t = 0; t < k; t++)
            {
                float s = alpha * B[t, j];
                if (s == 0f) continue;
                var aCol = A.GetColumn(t);
                MultiplyAddScalar(aCol, s, cCol);
            }
        }
    }

    // Out += alpha * u v^T (u, v — длины d и k соответственно; Out — d x k)
    public static void OuterAdd(Span<float> u, ReadOnlySpan<float> v, MatrixFloat Out, float alpha = 1f)
    {
        int d = Out.Dimensions[0];
        int k = Out.Dimensions[1];
        for (int j = 0; j < k; j++)
        {
            float s = alpha * v[j];
            if (s == 0f) continue;
            var outCol = Out.GetColumn(j);
            MultiplyAddScalar(u, s, outCol);
        }
    }

    public static void MultiplyAddScalar(ReadOnlySpan<float> a, float scalar, Span<float> dest)
    {
        TensorPrimitives.MultiplyAdd(a, scalar, dest, dest);        
    }

    public static void SubtractMatrices(MatrixFloat A, MatrixFloat B, MatrixFloat C) // C = A - B
    {
        TensorPrimitives.Subtract(A.Data, B.Data, C.Data);        
    }

    public static void AddMatricesInPlace(MatrixFloat A, MatrixFloat B) // A += B
    {
        TensorPrimitives.Add(A.Data, B.Data, A.Data);
    }

    public static void ScaleMatrixInPlace(MatrixFloat A, float s)
    {
        TensorPrimitives.Multiply(A.Data, s, A.Data);
    }

    public static void SetIdentity(MatrixFloat A)
    {
        A.Clear();
        int d = A.Dimensions[0];
        int n = A.Dimensions[1];
        int m = Math.Min(d, n);
        for (int i = 0; i < m; i++) A[i, i] = 1f;
    }

    // C = W^T * W (оба квадратные 300x300)
    public static void Gram(MatrixFloat W, MatrixFloat C)
    {
        int d = W.Dimensions[0];
        int k = W.Dimensions[1];
        if (d != k || C.Dimensions[0] != k || C.Dimensions[1] != k)
            throw new ArgumentException("Expect square matrices for Gram.");

        C.Clear();
        // C[j, t] = dot(W[:,j], W[:,t])
        for (int j = 0; j < k; j++)
        {
            var wj = W.GetColumn(j);
            for (int t = j; t < k; t++)
            {
                var wt = W.GetColumn(t);
                float dot = TensorPrimitives.Dot(wj, wt);
                C[j, t] += dot;
                if (t != j) C[t, j] += dot;
            }
        }
    }

    public static void AddIdentityInPlace(MatrixFloat A, float alpha = 1f)
    {
        int m = Math.Min(A.Dimensions[0], A.Dimensions[1]);
        for (int i = 0; i < m; i++) A[i, i] += alpha;
    }

    public static void SubIdentityInPlace(MatrixFloat A, float alpha = 1f)
    {
        AddIdentityInPlace(A, -alpha);
    }
}