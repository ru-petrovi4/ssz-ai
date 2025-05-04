﻿ using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using static Ssz.AI.Models.Cortex;
using Size = System.Drawing.Size;

namespace Ssz.AI.Models
{
    public class Model8
    {
        #region construction and destruction

        /// <summary>
        ///     Гистограммы для миниколонок
        /// </summary>
        public Model8()
        {
            Logger = ActivatorUtilities.CreateInstance<Logger<Model8>>(Program.Host.Services);
            DataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            (Labels, Images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            //GradientDistribution gradientDistribution = new();

            GradientMatricesCollection = new(Images.Length);
            foreach (int i in Enumerable.Range(0, Images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobelObsoslete(Images[i], MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
                GradientMatricesCollection.Add(gm);
                
                //SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            Retina = new Retina(Constants, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            //Retina.GenerateOwnedData(Constants, gradientDistribution);
            //Helpers.SerializationHelper.SaveToFile("retina.bin", Retina, null);
            Helpers.SerializationHelper.LoadFromFileIfExists("retina.bin", Retina, null);
            Retina.Prepare();

            Cortex = new Cortex(Constants, Retina);
            Cortex.GenerateOwnedData(Retina);            
            //Helpers.SerializationHelper.LoadFromFileIfExists(@"autoencoder.bin", Cortex, "autoencoder");
            Cortex.Prepare();
            //Helpers.SerializationHelper.SaveToFile(@"autoencoder.bin", Cortex, "autoencoder");

            Task.Factory.StartNew(() =>
            {
                LoadOrCalculateAutoencoder();

                //FindHyperColumn();
            }, TaskCreationOptions.LongRunning);
        }        

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();

        public readonly byte[] Labels;
        public readonly byte[][] Images;
        public readonly List<GradientInPoint[,]> GradientMatricesCollection;
        public int CurrentInputIndex = 0;

        public readonly Retina Retina;

        public readonly Cortex Cortex;        

        public int Generated_CenterX { get; set; }
        public int Generated_CenterXDelta { get; set; }
        public int Generated_CenterY { get; set; }
        public double Generated_AngleDelta { get; set; }
        public double Generated_Angle { get; set; }

        public CancellationTokenSource Temp_StopAutoencoderFinding_CancellationTokenSource  { get; set; } = new();

        public ILogger Logger { get; }

        public DataToDisplayHolder DataToDisplayHolder { get; }

        public Image[] GetImages1(double positionK, double angleK)
        {
            //(GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);

            //var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            //ActivitiyMaxInfo activitiyMaxInfo = new();

            ////GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            //List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            //foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
            //    foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
            //    {
            //        Detector d = Retina.Detectors[dx, dy];
            //        if (d.Temp_IsActivated)
            //            activatedDetectors.Add(d);
            //    }
            //var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
            //    Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
            //    Cortex.MiniColumns.Dimensions[0] / 2,
            //    Cortex.MiniColumns.Dimensions[1] / 2,
            //    Cortex.SubAreaMiniColumnsRadius + 2);            

            //return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
            return [];
        }

        private (GradientInPoint[,], Bitmap) GetGeneratedLine_gradientMatrix(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            Generated_CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0);
            Generated_CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + Generated_CenterXDelta;
            Generated_CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            Generated_AngleDelta = angleK * 2.0 * Math.PI;
            Generated_Angle = Math.PI / 2 + Generated_AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(Generated_CenterX + lineLength * Math.Cos(Generated_Angle));
            int endY = (int)(Generated_CenterY + lineLength * Math.Sin(Generated_Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(Generated_CenterX - lineLength * Math.Cos(Generated_Angle));
            int startY = (int)(Generated_CenterY - lineLength * Math.Sin(Generated_Angle));

            Bitmap originalBitmap = new Bitmap(Constants.GeneratedImageWidth, Constants.GeneratedImageHeight);
            using (Graphics g = Graphics.FromImage(originalBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Создаем кисть и устанавливаем толщину линии
                using (Pen pen = new Pen(Color.White, 15))
                {
                    // Рисуем наклонную линию
                    g.DrawLine(pen, startX, startY, endX, endY);
                }
            }

            // Уменьшаем изображение до размера 28x28

            // Создаем пустое изображение 28x28
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            using (Graphics g = Graphics.FromImage(resizedBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры для уменьшения
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Масштабируем изображение
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению            
            return (SobelOperator.ApplySobelObsoslete(resizedBitmap, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels), resizedBitmap);
        }

        public Image[] GetImages2()
        {
            //float[] inputHash = new float[Retina.Detectors.Data.Length];

            //float cosineSimilarity = 0.0f;

            //int calculateCountLocal = 0;
            //CurrentInputIndex = 100;
            //foreach (var _ in Enumerable.Range(0, 10))
            //{
            //    CurrentInputIndex += 1;

            //    var gradientMatrix = GradientMatricesCollection[CurrentInputIndex];

            //    Array.Clear(inputHash);

            //    Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: Cortex.SubArea_Detectors.Length,
            //        di =>
            //        {
            //            var d = Cortex.SubArea_Detectors[di];
            //            inputHash[di] = d.GetIsActivated(gradientMatrix) ? 1.0f : 0.0f;
            //        });

            //    float cs = Cortex.InputAutoencoder.Calculate(inputHash, learningRate: 0.01f);
            //    if (!float.IsNaN(cs))
            //    {
            //        cosineSimilarity += cs;
            //        calculateCountLocal += 1;
            //    }
            //    else
            //    {
            //    }
            //    //float sum = TensorPrimitives.Sum(autoencoder.Bottleneck.Buffer);
            //}

            //if (calculateCountLocal > 0)
            //    Cortex.InputAutoencoder.State_ControlCosineSimilarity = cosineSimilarity / calculateCountLocal;
            //else
            //    Cortex.InputAutoencoder.State_ControlCosineSimilarity = 0;

            //var image = Visualisation.GetContextSyncingMatrixFloatBitmap(DataToDisplayHolder.ContextSyncingMiniColumn?.Temp_ShortHashConversionMatrix, DataToDisplayHolder.ContextSyncingMiniColumn?.Temp_ShortHashConversionMatrix_TrainingCount);

            //var image = Visualisation.GetBigMatrixFloatBitmap(Cortex.InputAutoencoder.State_WeightsEncoder, 10, 100);

            //if (image is null)
            //    return [];
            //return [ image ];

            return [];
        }        

        public Image[] GetImages3()
        {
            var image = Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex);

            return [ image ];
        }

        public void CollectMemories_MNIST(int stepsCount)
        {
            DataToDisplayHolder.MiniColumsBitsCountInHashDistribution2 = new ulong[Constants.CortexWidth, Constants.CortexHeight, Constants.HashLength];

            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentInputIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentInputIndex];

                DoStep_CollectMemories_MNIST(gradientMatrix);
            }
        }

        public void DoStep_GeneratedLine(double positionK, double angleK)
        {
            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);
            
            //DoStep(gradientMatrix, activitiyMaxInfo, random);            
        }        

        #endregion

        private void LoadOrCalculateAutoencoder()
        {
            var cancellationToken = Temp_StopAutoencoderFinding_CancellationTokenSource.Token;

            const string fileName = @"autoencoder.bin";

            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, Cortex, "autoencoder");

            Logger.LogInformation($"CalculateAutoencoder(...) started;");

            Stopwatch sw = Stopwatch.StartNew();

            float[] input_Hash = new float[Retina.Detectors.Data.Length];

            Cortex.InputAutoencoder.State_CosineSimilarity = float.MaxValue;
            float cosineSimilarity;

            try
            {
                Cortex.InputAutoencoder.State_IterationsCount = 0;
                int stopIterationsCount = 0;
                while (stopIterationsCount < 5)
                {   
                    CurrentInputIndex = -1;
                    foreach (var _ in Enumerable.Range(0, 5000))
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (stopIterationsCount >= 5)
                            break;

                        CurrentInputIndex += 1;

                        var gradientMatrix = GradientMatricesCollection[CurrentInputIndex];

                        Array.Clear(input_Hash);

                        Stopwatch swLocal = Stopwatch.StartNew();

                        Parallel.For(
                            fromInclusive: 0,
                            toExclusive: Retina.Detectors.Data.Length,
                            di =>
                            {
                                var d = Retina.Detectors.Data[di];
                                input_Hash[di] = d.GetIsActivated_Obsolete(gradientMatrix, Constants) ? 1.0f : 0.0f;
                            });

                        cosineSimilarity = Cortex.InputAutoencoder.Calculate(input_Hash, learningRate: 0.01f);
                        if (!float.IsNaN(cosineSimilarity))
                        {
                            float cosineSimilarityDelta = cosineSimilarity - Cortex.InputAutoencoder.State_CosineSimilarity;
                            if (cosineSimilarityDelta > -0.0001f && cosineSimilarityDelta < 0.0001f)
                                stopIterationsCount += 1;
                            else
                                stopIterationsCount = 0;
                        }

                        Cortex.InputAutoencoder.State_CosineSimilarity = cosineSimilarity;
                        Cortex.InputAutoencoder.State_IterationsCount += 1;

                        swLocal.Stop();

                        Logger.LogInformation($"Image Processed: {CurrentInputIndex}; CosineSimilarity: {Cortex.InputAutoencoder.State_CosineSimilarity}; ElapsedMilliseconds: {swLocal.ElapsedMilliseconds}; Processed: {Cortex.InputAutoencoder.State_IterationsCount}");                        
                    }                    
                }
            }
            catch
            {
            }

            cosineSimilarity = 0.0f;
            int calculateCountLocal = 0;

            foreach (var _ in Enumerable.Range(0, 10))
            {
                CurrentInputIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentInputIndex];

                Array.Clear(input_Hash);

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Retina.Detectors.Data.Length,
                    di =>
                    {
                        var d = Retina.Detectors.Data[di];
                        input_Hash[di] = d.GetIsActivated_Obsolete(gradientMatrix, Constants) ? 1.0f : 0.0f;
                    });

                Cortex.InputAutoencoder.Calculate_ForwardPass(input_Hash);
                float cs = TensorPrimitives.CosineSimilarity(input_Hash, Cortex.InputAutoencoder.Temp_Output_Hash);
                if (!float.IsNaN(cs))
                {
                    cosineSimilarity += cs;
                    calculateCountLocal += 1;
                }
                else
                {
                }
                //float sum = TensorPrimitives.Sum(autoencoder.Bottleneck.Buffer);
            }

            if (calculateCountLocal > 0)
                Cortex.InputAutoencoder.State_ControlCosineSimilarity = cosineSimilarity / calculateCountLocal;
            else
                Cortex.InputAutoencoder.State_ControlCosineSimilarity = 0;

            sw.Stop();
            Cortex.InputAutoencoder.State_TrainingDurationMilliseconds = sw.ElapsedMilliseconds;

            Logger.LogInformation($"CalculateAutoencoder(...) finished; ControlCosineSimilarity: {Cortex.InputAutoencoder.State_ControlCosineSimilarity}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}; Processed: {Cortex.InputAutoencoder.State_IterationsCount}");

            if (Cortex.InputAutoencoder.State_IterationsCount > 0)
            {
                Helpers.SerializationHelper.SaveToFile(fileName, Cortex, "autoencoder");
            }
        }        

        private void FindHyperColumn()
        {
            MiniColumn winnerMiniColumn = Cortex.SubArea_MiniColumns[0];
            foreach (int mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
            {
                var mc = Cortex.SubArea_MiniColumns[mci];                
                mc.Temp_IsShortHashMustBeCalculated = false;
                if (mc.Memories.Count > winnerMiniColumn.Memories.Count)
                    winnerMiniColumn = mc;
            }

            winnerMiniColumn.Temp_IsSynced = true;
            ObjectManager<MiniColumn> syncedMiniColumnsToProcess = new ObjectManager<MiniColumn>(1000);
            winnerMiniColumn.Temp_SyncedMiniColumnsToProcess_Handle = syncedMiniColumnsToProcess.Add(winnerMiniColumn);
            winnerMiniColumn.ShortHashConversion = new int[Constants.ShortHashLength];
            foreach (int i in Enumerable.Range(0, Constants.ShortHashLength))
            {
                winnerMiniColumn.ShortHashConversion[i] = i; // No Conversion
            }

            DataToDisplayHolder.ContextSyncingMiniColumn = Cortex.MiniColumns[winnerMiniColumn.MCX, winnerMiniColumn.MCY - 1];

            // Кэш свободных матриц
            Stack<MatrixFloat> freeMatrixFloatsStack = new(100);

            // TEMPCODE
            //while (true)
            {
                CurrentInputIndex = -1; // Перед первым элементом
                foreach (var _ in Enumerable.Range(0, 5000))
                {
                    CurrentInputIndex += 1;

                    var gradientMatrix = GradientMatricesCollection[CurrentInputIndex];

                    foreach (MiniColumn miniColumn in syncedMiniColumnsToProcess.ToArray())
                    {
                        bool anyToProcessNearestMiniColumn = false;
                        //foreach (var nearestMiniColumn in miniColumn.NearestMiniColumnInfos[0])
                        //{
                        //    if (!nearestMiniColumn.Temp_IsSynced)
                        //    {
                        //        nearestMiniColumn.Temp_IsShortHashMustBeCalculated = true;
                        //        anyToProcessNearestMiniColumn = true;
                        //    }
                        //}
                        if (!anyToProcessNearestMiniColumn)
                        {
                            syncedMiniColumnsToProcess.Remove(miniColumn.Temp_SyncedMiniColumnsToProcess_Handle);
                        }
                    }

                    if (syncedMiniColumnsToProcess.Count == 0)
                        break;

                    for (int di = 0; di < Cortex.SubArea_Detectors.Length; di += 1)
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.Temp_IsActivated = d.GetIsActivated_Obsolete(gradientMatrix, Constants);
                    }
                    //Parallel.For(
                    //    fromInclusive: 0,
                    //    toExclusive: Cortex.SubArea_Detectors.Length,
                    //    di =>
                    //    {
                    //        var d = Cortex.SubArea_Detectors[di];
                    //        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    //    });

                    foreach (MiniColumn miniColumn in syncedMiniColumnsToProcess.ToArray())
                    {
                        miniColumn.GetHash(miniColumn.Temp_Hash);
                        if (TensorPrimitives.Sum(miniColumn.Temp_Hash) < Constants.MinBitsInHashForMemory)
                            continue;
                        miniColumn.Autoencoder!.Calculate_ForwardPass(miniColumn.Temp_Hash);
                        miniColumn.GetShortHashConverted(miniColumn.Autoencoder!.Temp_ShortHash, miniColumn.Temp_ShortHashConverted);

                        //foreach (var nearestMiniColumn in miniColumn.NearestMiniColumnInfos[0])
                        //{
                        //    if (!nearestMiniColumn.Temp_IsSynced)
                        //    {
                        //        if (nearestMiniColumn.Temp_IsShortHashMustBeCalculated)
                        //        {
                        //            nearestMiniColumn.Temp_IsShortHashMustBeCalculated = false;

                        //            nearestMiniColumn.GetHash(nearestMiniColumn.Temp_Hash);
                        //            if (TensorPrimitives.Sum(nearestMiniColumn.Temp_Hash) < Constants.MinBitsInHashForMemory)
                        //                continue;
                        //            nearestMiniColumn.Autoencoder!.Calculate_ForwardPass(nearestMiniColumn.Temp_Hash);
                        //        }
                        //        else
                        //        {
                        //            if (TensorPrimitives.Sum(nearestMiniColumn.Temp_Hash) < Constants.MinBitsInHashForMemory)
                        //                continue;
                        //        }

                        //        if (nearestMiniColumn.Temp_ShortHashConversionMatrix is null)
                        //        {
                        //            if (freeMatrixFloatsStack.Count == 0)
                        //            {
                        //                nearestMiniColumn.Temp_ShortHashConversionMatrix = new MatrixFloat(Constants.ShortHashLength, Constants.ShortHashLength);                                        
                        //            }
                        //            else
                        //            {
                        //                var freeMatrixFloat = freeMatrixFloatsStack.Pop();
                        //                Array.Clear(freeMatrixFloat.Data);
                        //                nearestMiniColumn.Temp_ShortHashConversionMatrix = freeMatrixFloat;
                        //            }
                        //            nearestMiniColumn.Temp_ShortHashConversionMatrix_TrainingCount = 0;
                        //        }

                        //        nearestMiniColumn.Temp_Hash_SyncQualitySum += TensorPrimitives.CosineSimilarity(nearestMiniColumn.Temp_Hash, miniColumn.Temp_Hash);
                        //        nearestMiniColumn.Temp_Hash_SyncQualitySumCount += 1;
                        //        nearestMiniColumn.Temp_Hash_SyncQuality = nearestMiniColumn.Temp_Hash_SyncQualitySum / nearestMiniColumn.Temp_Hash_SyncQualitySumCount;

                        //        nearestMiniColumn.Temp_ShortHash_AutoencoderSyncQualitySum += TensorPrimitives.CosineSimilarity(nearestMiniColumn.Temp_Hash, nearestMiniColumn.Autoencoder!.Temp_Output_Hash);
                        //        nearestMiniColumn.Temp_ShortHash_AutoencoderSyncQualitySumCount += 1;
                        //        nearestMiniColumn.Temp_ShortHash_AutoencoderSyncQuality = nearestMiniColumn.Temp_ShortHash_AutoencoderSyncQualitySum / nearestMiniColumn.Temp_ShortHash_AutoencoderSyncQualitySumCount;

                        //        bool synced = MiniColumnsSyncronization.TrainSyncronization(nearestMiniColumn, miniColumn.Temp_ShortHashConverted); // nearestMiniColumn.Temp_ShortHash miniColumn.Temp_ShortHashConverted
                        //        if (synced)
                        //        {
                        //            nearestMiniColumn.Temp_IsSynced = true;
                        //            nearestMiniColumn.Temp_SyncedMiniColumnsToProcess_Handle = syncedMiniColumnsToProcess.Add(nearestMiniColumn);
                                    
                        //            // TEMPCODE
                        //            //freeMatrixFloatsStack.Push(nearestMiniColumn.Temp_ShortHashConversionMatrix);
                        //            //nearestMiniColumn.Temp_ShortHashConversionMatrix = null;
                        //        }
                        //    }
                        //}
                    }
                }
            }
        }

        private void DoStep_CollectMemories_MNIST(GradientInPoint[,] gradientMatrix)
        {
            //for (int di = 0; di < Cortex.SubArea_Detectors.Length; di += 1)
            //{
            //    var d = Cortex.SubArea_Detectors[di];
            //    d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
            //}
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.Temp_IsActivated = d.GetIsActivated_Obsolete(gradientMatrix, Constants);
                    });

            //for (int mci = 0; mci < Cortex.SubArea_MiniColumns.Length; mci += 1)
            //{
            //    var mc = Cortex.SubArea_MiniColumns[mci];
            //    mc.CalculateHash(mc.Temp_Hash);

            //    int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);
            //    DataToDisplayHolder.MiniColumsBitsCountInHashDistribution2[mc.MCX, mc.MCY, bitsCountInHash] += 1;

            //    if (bitsCountInHash >= Constants.MinBitsInHashForMemory)
            //    {
            //        mc.Memories.Add(new Memory { Hash = (float[])mc.Temp_Hash.Clone() });
            //    }
            //}
            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.GetHash(mc.Temp_Hash);

                    int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);
                    DataToDisplayHolder.MiniColumsBitsCountInHashDistribution2[mc.MCX, mc.MCY, bitsCountInHash] += 1;

                    if (bitsCountInHash >= Constants.MinBitsInHashForMemory)
                    {
                        mc.AddMemory(new Memory { Hash = (float[])mc.Temp_Hash.Clone() });
                    }
                });
        }        

        private Autoencoder FindAutoencoder(MiniColumn miniColumn)
        {
            Stopwatch sw = Stopwatch.StartNew();

            var autoencoder = new Autoencoder();
            autoencoder.GenerateOwnedData(inputSize: Constants.HashLength, bottleneckSize: Constants.ShortHashLength, bottleneck_MaxBitsCount: Constants.ShortHashBitsCount);
            autoencoder.Prepare();

            autoencoder.State_CosineSimilarity = float.MaxValue;
            float cosineSimilarity = 1.0f;
            float cosineSimilarityDelta = 1.0f;

            int trainCount = (int)(miniColumn.Memories.Count * 0.9);
            int memoriesCount = 0;

            autoencoder.State_IterationsCount = 0;
            int stopIterationsCount = 0;
            while (stopIterationsCount < 5)
            {
                cosineSimilarity = 0.0f;
                
                memoriesCount = 0;
                foreach (var memory in miniColumn.Memories.Take(trainCount))
                {
                    var input = memory.Hash;

                    float cs = autoencoder.Calculate(input, learningRate: 0.01f);
                    if (!float.IsNaN(cs))
                    {
                        cosineSimilarity += cs;
                        memoriesCount += 1;
                    }
                    else
                    {
                    }
                }

                if (memoriesCount > 0)
                    cosineSimilarity = cosineSimilarity / memoriesCount;

                cosineSimilarityDelta = cosineSimilarity - autoencoder.State_CosineSimilarity;
                if (cosineSimilarityDelta > -0.0001f && cosineSimilarityDelta < 0.0001f)
                    stopIterationsCount += 1;
                else
                    stopIterationsCount = 0;

                autoencoder.State_IterationsCount += 1;

                autoencoder.State_CosineSimilarity = cosineSimilarity;                
            }

            cosineSimilarity = 0.0f;

            memoriesCount = 0;
            foreach (var memory in miniColumn.Memories.Skip(trainCount))
            {
                var input_Hash = memory.Hash;

                autoencoder.Calculate_ForwardPass(input_Hash);

                float cs = TensorPrimitives.CosineSimilarity(input_Hash, autoencoder.Temp_Output_Hash);                
                if (!float.IsNaN(cs))
                {
                    cosineSimilarity += cs;
                    memoriesCount += 1;
                }
                else
                {
                }
                //float sum = TensorPrimitives.Sum(autoencoder.Bottleneck.Buffer);
            }

            if (memoriesCount > 0)
                autoencoder.State_ControlCosineSimilarity = cosineSimilarity / memoriesCount;
            else
                autoencoder.State_ControlCosineSimilarity = 0;

            sw.Stop();
            autoencoder.State_TrainingDurationMilliseconds = sw.ElapsedMilliseconds;            

            return autoencoder;
        }        

        public static readonly Color[] DefaultColors =
        {
            Color.FromArgb(0xFF, 0x00, 0xFE),
            Color.FromArgb(0x02, 0x00, 0xF9),
            Color.FromArgb(0x00, 0xFF, 0xFF),
            Color.FromArgb(0xFF, 0x80, 0x41),
            Color.FromArgb(0xFC, 0x01, 0x00),
            Color.FromArgb(0x00, 0xFF, 0x01),
            Color.FromArgb(0xFF, 0xFF, 0x00),
            Color.FromArgb(0xFF, 0x00, 0x00),            
        };

        private class VisualizationTableItemsCluster
        {
            public float[] Hash = null!;

            public float[] TempHash = null!;

            public List<VisualizationTableItem> VisualizationTableItems = null!;

            public float MinCosineSimilarity;

            public float MaxCosineSimilarity;            
        }

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {
            /// <summary>
            ///     Расстояние между детекторами по горизонтали и вертикали  
            ///     [0..MNISTImageWidth]
            /// </summary>
            public double DetectorDelta => 0.1;

            public int AngleRangeDegree_LimitMagnitude => 300;

            public double DetectorMinGradientMagnitude => 5;

            public int GeneratedMinGradientMagnitude => 5;

            public int GeneratedMaxGradientMagnitude => 1200;

            public int AngleRangeDegreeMin => 60;

            public int AngleRangeDegreeMax => 60;

            public int MagnitudeRangesCount => 4;

            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidthPixels => MNISTHelper.MNISTImageWidthPixels;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeightPixels => MNISTHelper.MNISTImageHeightPixels;            

            public int GeneratedImageWidth => 280;

            public int GeneratedImageHeight => 280;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси X
            /// </summary>
            public int CortexWidth => 200;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси Y
            /// </summary>
            public int CortexHeight => 200;            

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;            

            public int HashLength => 200;

            public int ShortHashLength => 50;

            public int ShortHashBitsCount => 11;            

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 11;           

            /// <summary>
            ///     Примерное количество воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 400;

            /// <summary>
            ///     Индекс X центра подобласти [0..CortexWidth]
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти [0..CortexHeight]
            /// </summary>
            public int SubAreaCenter_Cy => 100;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int MiniColumnsMaxDistance => 1;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public float MemoryClustersThreshold => 0.66f;

            public int Angle_SmallPoints_Count => 1000;

            public float Angle_SmallPoints_Radius => 0.003f;

            public int Angle_BigPoints_Count => 200;

            public float Angle_BigPoints_Radius => 0.015f;
        }        
    }
}


//private void FindAutoencoder(MiniColumn miniColumn)
//{
//    // Параметры модели
//    int inputSize = 200;
//    int hiddenSize = 50;
//    int batchSize = 32;
//    int epochs = 20;

//    // Создание весов и смещений
//    var weights_encoder = tf.Variable(tf.random.normal((inputSize, hiddenSize)), name: "weights_encoder", trainable: true);
//    var biases_encoder = tf.Variable(tf.zeros(hiddenSize), name: "biases_encoder", trainable: true);

//    var weights_decoder = tf.Variable(tf.random.normal((hiddenSize, inputSize)), name: "weights_decoder", trainable: true);
//    var biases_decoder = tf.Variable(tf.zeros(inputSize), name: "biases_decoder", trainable: true);

//    if (weights_encoder == null || biases_encoder == null || weights_decoder == null || biases_decoder == null)
//    {
//        Console.WriteLine("Ошибка: один из параметров модели не инициализирован.");
//        return;
//    }

//    // Функция прямого прохода
//    Func<NDArray, NDArray> forward = input =>
//    {
//        //var encoded = tf.sigmoid(tf.matmul(input, weights_encoder) + biases_encoder);
//        //var decoded = tf.sigmoid(tf.matmul(encoded, weights_decoder) + biases_decoder);
//        //return decoded.numpy();

//        var inputTensor = tf.convert_to_tensor(input);
//        var encoded = tf.sigmoid(tf.matmul(inputTensor, weights_encoder) + biases_encoder);
//        var decoded = tf.sigmoid(tf.matmul(encoded, weights_decoder) + biases_decoder);
//        return decoded.numpy();
//    };

//    // Функция потерь
//    Func<NDArray, NDArray, Tensor> loss_fn = (inputs, outputs) =>
//    {
//        //return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: inputs, logits: outputs));

//        var inputsTensor = tf.convert_to_tensor(inputs);
//        var outputsTensor = tf.convert_to_tensor(outputs);
//        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: inputsTensor, logits: outputsTensor));
//    };

//    // Оптимизатор
//    var optimizer = tf.keras.optimizers.Adam(learning_rate: 0.001f);

//    // Генерация данных для тренировки
//    var trainData = GenerateBooleanData(10000, inputSize, 15);

//    // Обучение модели
//    for (int epoch = 0; epoch < epochs; epoch++)
//    {
//        foreach (var batch in GetBatches(trainData, batchSize))
//        {
//            using (var tape = tf.GradientTape())
//            {
//                tape.watch(weights_encoder);
//                tape.watch(biases_encoder);
//                tape.watch(weights_decoder);
//                tape.watch(biases_decoder);

//                var outputs = forward(batch);
//                var loss = loss_fn(batch, outputs);

//                if (loss == null)
//                {
//                    Console.WriteLine("Ошибка: Loss не был рассчитан.");
//                    return;
//                }

//                tape.watch(loss);

//                // Вычисление и применение градиентов
//                var gradients = tape.gradient(loss, new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder });
//                if (gradients == null || gradients.Contains(null))
//                {
//                    Console.WriteLine("Ошибка: градиенты не были рассчитаны.");
//                    return;
//                }

//                var zipped = gradients.Zip(new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder });
//                optimizer.apply_gradients(zipped); // zip(gradients, new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder })
//            }
//        }

//        // Вывод текущей ошибки
//        var epochLoss = loss_fn(trainData, forward(trainData));
//        Console.WriteLine($"Эпоха {epoch + 1}/{epochs}, Потеря: {epochLoss.numpy()}");
//    }
//}

//static NDArray GenerateBooleanData(int samples, int size, int onesPerVector)
//{
//    Random rnd = new Random();
//    var data = np.zeros((samples, size));

//    for (int i = 0; i < samples; i++)
//    {
//        var indices = new int[size];
//        for (int j = 0; j < size; j++)
//            indices[j] = j;

//        for (int j = 0; j < onesPerVector; j++)
//        {
//            int index = rnd.Next(size - j);
//            data[i, indices[index]] = 1.0;
//            indices[index] = indices[size - j - 1];
//        }
//    }

//    return data.astype(np.float32);
//}

//static IEnumerable<NDArray> GetBatches(NDArray data, int batchSize)
//{
//    for (int i = 0; i < data.shape[0]; i += batchSize)
//    {
//        yield return data[$"{i}:{Math.Min(i + batchSize, data.shape[0])}"];
//    }
//}

//foreach (int mcy in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[1]))
//    foreach (int mcx in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[0]))
//    {
//        var mc = Cortex.MiniColumns[mcx, mcy];
//        mc.Temp_IsShortHashMustBeCalculated = false;
//        if (mc.Memories.Count > winnerMiniColumn.Memories.Count)
//            winnerMiniColumn = mc;
//    }
