using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model01
    {
        //public void CalculatePrimaryWords_Em(ILoggersSet loggersSet)
        //{ 
        //    var totalStopwatch = Stopwatch.StartNew();            

        //    var r = new Random();            

        //    // Original vector length.
        //    const int q = OldVectorLength;
        //    // Clusters count.
        //    const int k = 1000;
        //    // Words count.
        //    int n = Words_RU.Count;                       

        //    // Математические ожидания
        //    Matrix<float> matrixM = DenseMatrix.Build.Dense(q, k);
        //    // Ковариации для k-го кластера
        //    List<Matrix<double>> matricesR_List = new(k);          
        //    // Веса кластеров
        //    Matrix<float> matrixW = DenseMatrix.Build.Dense(1, k);
        //    // Матрица нормированных вероятностей принадлежности наблюдений к кластерам.
        //    Matrix<float> matrixX = DenseMatrix.Build.Dense(n, k);
        //    Matrix<float> matrixTempX_row = DenseMatrix.Build.Dense(1, k);
        //    Matrix<float> matrixTempSumX_row = DenseMatrix.Build.Dense(1, k);
        //    // y vector
        //    Matrix<float> matrixY = DenseMatrix.Build.Dense(q, 1);                                             
        //    // Temp qxk
        //    Matrix<float> matrixTemp_q_k = DenseMatrix.Build.Dense(q, k);            
        //    // Dispersion vector
        //    Matrix<double> matrix_d2 = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Dense(n, k);
        //    // Dispersion vector
        //    Matrix<double> matrix_ln_p = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Dense(1, k);
        //    // Минимальное приращение функции логарифмического правдоподобия по достижении которого алгоритм останавливается
        //    const double E = Double.MinValue;
        //    // Максимальное число итераций алгоритма (если алгоритм не сходится, то будет остановлен при достижении данного числа итераций)
        //    const int Q_MAX = Int32.MaxValue;
        //    const float W_Initial = 1.0f / k;

        //    double q_ln_2Pi = q * Math.Log(2 * Math.PI);

        //    #region Initialization

        //    for (int j = 0; j < k; j += 1)
        //    {
        //        int wordIndex = r.Next(Words_RU.Count);
        //        Word word = Words_RU[wordIndex];
        //        var oldVectror = word.OldVector;
        //        for (int t = 0; t < q; t += 1)
        //        {
        //            matrixM[t, j] = oldVectror[t];
        //        }

        //        matrixW[0, j] = W_Initial;

        //        matricesR_List.Add(MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Diagonal(q, q, 0.01));
        //    }

        //    //for (t = 0; t < q; t += 1)
        //    //{
        //    //    matrixR[t, t] = 1.0f;
        //    //}

        //    #endregion

        //    double delta_llh = Double.MaxValue;
        //    double llh;
        //    int Q = 0;
            
        //    //List<double> divisor_List = new(k);

        //    while (delta_llh > E && Q < Q_MAX)
        //    {
        //        var stopwatch = Stopwatch.StartNew();
        //        Q += 1;

        //        #region ЕXPECTATION

        //        //divisor_List.Clear();
        //        //for (int j = 0; j < k; j += 1)
        //        //{
        //        //    var matrixR = matricesR_List[j];
        //        //    double divisor = 1.0;

        //        //    for (int t = 0; t < q; t += 1)
        //        //    {
        //        //        divisor *= Math.Pow(Constants.Pi2 * matrixR[t, t], 0.5);
        //        //    }

        //        //    divisor_List.Add(divisor);
        //        //}

        //        matrixTempSumX_row.Clear();

        //        llh = 0;
        //        for (int i = 1; i < n; i += 1)
        //        {
        //            Word word = Words_RU[i];
        //            var oldVectror = word.OldVector;                                       
        //            double ln_sum_p = 0;                    
        //            for (int j = 0; j < k; j += 1)
        //            {
        //                var matrixR = matricesR_List[j];

        //                //// Not optimized version
        //                //double d2 = 0.0f;
        //                //for (int t = 0; t < q; t += 1)
        //                //{
        //                //    double delta = oldVectror[t] - matrixM[t, j];
        //                //    d2 += delta * delta / matrixR[t, t];
        //                //}
        //                //matrix_d2[i, j] = d2;
        //                //double p = (matrixW[0, j] / divisor_List[j]) * Math.Exp(-0.5 * d2);

        //                double ln_p = Math.Log(matrixW[0, j]) - 0.5 * q_ln_2Pi;
        //                for (int t = 0; t < q; t += 1)
        //                {                            
        //                    double delta = oldVectror[t] - matrixM[t, j];
        //                    double d2 = delta * delta / matrixR[t, t];
        //                    ln_p += -0.5 * (Math.Log(matrixR[t, t]) + d2);
        //                }                        
        //                matrix_ln_p[0, j] = ln_p;
        //                if (j == 0)
        //                    ln_sum_p = ln_p;
        //                else
        //                    ln_sum_p += Math.Log(1 + Math.Exp(ln_p - ln_sum_p));
        //            }
        //            for (int j = 0; j < k; j += 1)
        //            {
        //                //// Not optimized version
        //                //matrixX[i, j] = matrixTempX_row[0, j] = (float)(matrix_p[0, j] / sum_p);
        //                matrixX[i, j] = matrixTempX_row[0, j] = (float)Math.Exp(matrix_ln_p[0, j] - ln_sum_p);
        //            }

        //            // TEMPCOD
        //            double sum_temp = 0;
        //            for (int j = 0; j < k; j += 1)
        //            {                        
        //                sum_temp += matrixTempX_row[0, j];
        //            }

        //            delta_llh = ln_sum_p;
        //            llh += delta_llh;
        //            for (int t = 0; t < q; t += 1)
        //            {
        //                matrixY[t, 0] = oldVectror[t];
        //            }
        //            matrixY.Multiply(matrixTempX_row, matrixTemp_q_k);
        //            matrixM.Add(matrixTemp_q_k, matrixM);

        //            matrixTempSumX_row.Add(matrixTempX_row, matrixTempSumX_row);
        //        }

        //        #endregion

        //        stopwatch.Stop();
        //        loggersSet.UserFriendlyLogger.LogInformation("ЕXPECTATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        //        stopwatch.Restart();

        //        #region MAXIMIZATION                

        //        for (int j = 0; j < k; j += 1)
        //        {
        //            float w_j = matrixTempSumX_row[0, j];
        //            for (int t = 0; t < q; t += 1)
        //            {
        //                matrixM[t, j] = matrixM[t, j] / w_j;
        //            }
        //            matrixW[0, j] = w_j / (n - 1);

        //            var matrixR = matricesR_List[j];
        //            matrixR.Clear();

        //            //// Not optimized version
        //            //for (i = 1; i < n; i += 1)
        //            //{
        //            //    Word word = Words[i];                        
        //            //    for (int t = 0; t < q; t += 1)
        //            //    {
        //            //        var delta = word.OriginalNormalizedVector[t] - matrixM[t, j];
        //            //        matrixR[t, t] += delta * delta * matrixX[i, j];
        //            //    }
        //            //}

        //            Parallel.For(
        //                fromInclusive:1,
        //                toExclusive:Words_RU.Count,                        
        //                () => new float[k], // method to initialize the local variable
        //                (i, loopState, localMatrixR) => // method invoked by the loop on each iteration
        //                {
        //                    Word word = Words_RU[i];
        //                    for (int t = 0; t < q; t += 1)
        //                    {
        //                        var delta = word.OldVector[t] - matrixM[t, j];
        //                        localMatrixR[t] += delta * delta * matrixX[i, j];
        //                    }
        //                    return localMatrixR; // value to be passed to next iteration
        //                },
        //                localMatrixR => // Method to be executed when each partition has completed.
        //                {
        //                    lock (matrixR) // For thread-safety
        //                    {
        //                        for (int t = 0; t < q; t += 1)
        //                        {                                    
        //                            matrixR[t, t] += localMatrixR[t];
        //                        }
        //                    }
        //                });                    

        //            matrixR.Divide(n - 1, matrixR);
        //        }

        //        #endregion

        //        stopwatch.Stop();
        //        loggersSet.UserFriendlyLogger.LogInformation("MAXIMIZATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        //    }            

        //    totalStopwatch.Stop();
        //    loggersSet.UserFriendlyLogger.LogInformation("CalculateAlgorithmData_Em.PrimaryWords totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        //}        
    }    
}