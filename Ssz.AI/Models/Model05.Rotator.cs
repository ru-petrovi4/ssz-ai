#define CALC_BITS_COUNT_IN_HASH_HISTOGRAM

using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.AI.Views;
using Ssz.Utils;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using static Ssz.AI.Models.Cortex;
using Size = System.Drawing.Size;

namespace Ssz.AI.Models
{
    public class Model05
    {
        #region construction and destruction

        /// <summary>
        ///     Построение "вертушки"
        /// </summary>
        public Model05(ModelConstants constants)
        {
            Constants = constants;

            UserFriendlyLogger = new UserFriendlyLogger(DebugWindow.AddLine);

#if CALC_BITS_COUNT_IN_HASH_HISTOGRAM
            DataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();            
#endif

            Stopwatch sw = Stopwatch.StartNew();            

            var (Labels, Images) = (new byte[60000], new byte[60000][]);

            GradientDistribution? gradientDistribution = new();

            Random random = new(6);
            var t = sw.ElapsedMilliseconds;

            MonoInput = new MonoInput();
            MonoInput.GenerateOwnedData_Simplified2(
                random,
                Constants,
                gradientDistribution,
                Labels,
                Images);
            //SerializationHelper.LoadFromFileIfExists("MonoInput.bin", MonoInput, null);
            MonoInput.Prepare();
            //SerializationHelper.SaveToFile("MonoInput.bin", MonoInput, null);                  

            bool generateRetina = true;
            Retina = new Retina(Constants, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            if (generateRetina)
                Retina.GenerateOwnedData(random, Constants, gradientDistribution);
            if (!generateRetina)
                SerializationHelper.LoadFromFileIfExists("Retina.bin", Retina, null);
            Retina.Prepare();
            if (generateRetina)
                SerializationHelper.SaveToFile("Retina.bin", Retina, null);

            Cortex = new Cortex(Constants, Retina);
            Cortex.GenerateOwnedData(Retina);
            Cortex.Prepare();

            DetectorsActivationHash = new float[Constants.HashLength];
            GetImageWithDescs1(0.0, 0.0);
            DetectorsActivationHash0 = (float[])DetectorsActivationHash.Clone();
        }

        #endregion

        #region public functions

        public ILogger UserFriendlyLogger { get; }

        public DataToDisplayHolder DataToDisplayHolder = null!;

        public readonly ModelConstants Constants;        

        public MonoInput MonoInput { get; set; } = null!;

        public ActivitiyMaxInfo Temp_ActivitiyMaxInfo { get; } = new();        

        public float[] DetectorsActivationHash0 { get; set; }
        public float[] DetectorsActivationHash { get; set; }

        public int CurrentInputIndex = 0;

        public readonly Retina Retina;

        public readonly Cortex Cortex;        

        public int Generated_CenterX { get; set; }
        public int Generated_CenterXDelta { get; set; }
        public int Generated_CenterY { get; set; }
        public double Generated_AngleDelta { get; set; }
        public double Generated_Angle { get; set; }                

        public void ResetMemories()
        {
            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.Memories.Clear();
                });

#if CALC_BITS_COUNT_IN_HASH_HISTOGRAM
            Array.Clear(DataToDisplayHolder.MiniColumsBitsCountInHashDistribution);
#endif
        }

        public async Task DoSteps_MNISTAsync(int stepsCount, Random random, bool randomInitialization, bool reorderMemoriesPeriodically)
        {
            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentInputIndex += 1;

                var gradientMatrix = MonoInput.MonoInputItems[CurrentInputIndex].GradientMatrix;

                await DoStepAsync(CurrentInputIndex, gradientMatrix, Temp_ActivitiyMaxInfo, random, randomInitialization, reorderMemoriesPeriodically);
            }

            UserFriendlyLogger.LogInformation($"DoSteps() finished. stepsCount: {stepsCount}");
        }

        public void Flood(Random random, float floodRadius)
        {
            Cortex.MiniColumn maxMemoryMiniColumn = Cortex.CenterMiniColumn!;
            foreach (int mcy in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = Cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && mc.Memories.Count > 0)
                    {
                        if (mc.Memories.Count > maxMemoryMiniColumn.Memories.Count)
                            maxMemoryMiniColumn = mc;                        
                    }
                }

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    float r = MathF.Sqrt((mc.MCX - maxMemoryMiniColumn.MCX) * (mc.MCX - maxMemoryMiniColumn.MCX) +
                        (mc.MCY - maxMemoryMiniColumn.MCY) * (mc.MCY - maxMemoryMiniColumn.MCY));
                    if (r > floodRadius)
                        mc.Memories.Clear();
                });

            UserFriendlyLogger.LogInformation($"Flood() finished. floodRadius: {floodRadius}");
        }

        public void DoStep_GeneratedLine(double positionK, double angleK)
        {
            //var random = new Random();

            //ActivitiyMaxInfo activitiyMaxInfo = new();

            //(GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);
            
            //DoStep(-1, gradientMatrix, activitiyMaxInfo, random);            
        }

        public VisualizationWithDesc[] GetImageWithDescs1(double positionK, double angle)
        {
            (DenseMatrix<GradientInPoint> gradientMatrix, var resizedBitmap) = GetGeneratedLine_GradientMatrix(positionK, angle);

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            ActivitiyMaxInfo activitiyMaxInfo = new();

            CalculateDetectorsAndActivityAndSuperActivity(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            Cortex.CenterMiniColumn!.GetHash(DetectorsActivationHash);

            var activityColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColums_ActivityColor(Cortex),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);

            var superActivityColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColums_SuperActivityColor(Cortex, activitiyMaxInfo),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);            

            return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(gradientBitmap),
                    Desc = @"Полная картина градиентов" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(detectorsActivationBitmap),
                    Desc = @"Активация детекторов" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(activityColorImage),
                    Desc = @"Активность миниколонок (белый - максимум)" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(superActivityColorImage),
                    Desc = @"Суперактивность миниколонок (белый - максимум)" }
                ];
        }

        public VisualizationWithDesc[] GetImageWithDescs2()
        {
            int currentInputIndex;
            if (CurrentInputIndex < 0)
                currentInputIndex = 0;
            else
                currentInputIndex = CurrentInputIndex;

            MonoInputItem monoInputItem = MonoInput.MonoInputItems[currentInputIndex];

            var gradientMatrix = monoInputItem.GradientMatrix;
            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);
            var subImage = BitmapHelper.GetSubBitmap(
                gradientBitmap,
                (int)(Cortex.CenterMiniColumn!.CenterX * 10),
                (int)(Cortex.CenterMiniColumn!.CenterY * 10),
                (int)(Cortex.DetectorsVisibleRadius * 10));

            var activatedDetectors = Cortex.SubArea_Detectors.Where(d => d.Temp_IsActivated).ToList();
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var activityColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColums_ActivityColor(Cortex),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);

            var superActivityColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColums_SuperActivityColor(Cortex, null),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);            

            var memoriesColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);

            var memoriesCountImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubArea_MiniColumns_Radius + 2);            

            return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(subImage), 
                    Desc = $"Видимая картина градиентов. {monoInputItem.Label}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(detectorsActivationBitmap),
                    Desc = $"Активация детекторов. Activated Detectors: {activatedDetectors.Count}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(activityColorImage), 
                    Desc = @"Активность миниколонок (белый - максимум)" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(superActivityColorImage), 
                    Desc = @"Суперактивность миниколонок (белый - максимум)" },
                new Model3DWithDesc { Data = Visualization3D.GetSubArea_MiniColumnsMemories_Model3DScene(Cortex),
                    Desc = @"Накопленные воспоминания в миниколонках" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(memoriesColorImage),
                    Desc = @"Средний цвет накопленных воспоминаний в миниколонках" },                
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(memoriesCountImage), 
                    Desc = @"Количество воспоминаний в миниколонках" }
                ];
        }

        //public Image[] GetImages3()
        //{
        //    //var totalMnistBitmap = GetMnistTotalBitmap();

        //    var gradientMatrix = MonoInput.MonoInputItems[CurrentInputIndex].GradientMatrix;

        //    var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

        //    ActivitiyMaxInfo activitiyMaxInfo = new();

        //    //GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

        //    List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
        //    foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
        //        foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
        //        {
        //            Detector d = Retina.Detectors[dx, dy];
        //            if (d.Temp_IsActivated)
        //                activatedDetectors.Add(d);
        //        }
        //    var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

        //    var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
        //        Visualisation.GetMiniColumsActivityBitmap_Obsolete(Cortex, activitiyMaxInfo),
        //        Cortex.MiniColumns.Dimensions[0] / 2,
        //        Cortex.MiniColumns.Dimensions[1] / 2,
        //        Cortex.SubArea_MiniColumns_Radius + 2);
        //    //var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

        //    var originalBitmap = MNISTHelper.GetBitmap(
        //        MonoInput.MonoInputItems[CurrentInputIndex].Original_Image,
        //        MNISTHelper.MNISTImageWidthPixels,
        //        MNISTHelper.MNISTImageHeightPixels);

        //    return [originalBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        //}        

        public void GenerateRotator(Random random)
        {
            foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
            {
                int inputIndex = MonoInput.Images.Length + mci;
                if (inputIndex >= MonoInput.MonoInputItems.Length)
                    continue;

                MiniColumn winnerMiniColumn = Cortex.SubArea_MiniColumns[mci];

                winnerMiniColumn.Temp_Memories.Clear();

                var dx = winnerMiniColumn.MCX - Cortex.CenterMiniColumn!.MCX;
                var dy = winnerMiniColumn.MCY - Cortex.CenterMiniColumn!.MCY;

                double magnitude = Constants.GeneratedMinGradientMagnitude + 
                    (Constants.GeneratedMaxGradientMagnitude - Constants.GeneratedMinGradientMagnitude) * Math.Sqrt(dx * dx + dy * dy) / Cortex.SubArea_MiniColumns_Radius;
                double angle = Math.Atan2(dy, dx);

                int gradX = (int)(Math.Cos(angle) * magnitude);
                int gradY = (int)(Math.Sin(angle) * magnitude);

                GradientInPoint gradientInPoint = new()
                {
                    GradX = gradX,
                    GradY = gradY,
                    Magnitude = magnitude,
                    Angle = angle,
                };

                int width = MNISTHelper.MNISTImageWidthPixels;
                int height = MNISTHelper.MNISTImageHeightPixels;
                var generatedGradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

                for (int y = 1; y < height - 1; y += 1)
                {
                    for (int x = 1; x < width - 1; x += 1)
                    {
                        generatedGradientMatrix[x, y] = gradientInPoint;
                    }
                }

                ActivitiyMaxInfo activitiyMaxInfo = new();

                CalculateDetectorsAndActivityAndSuperActivity(generatedGradientMatrix, activitiyMaxInfo);
                
                MonoInputItem monoInputItem = new();
                monoInputItem.Label = $"Maginitude: {(int)magnitude}; Angle: {(int)MathHelper.RadiansToDegrees(angle)}";
                //monoInputItem.Original_Image = original_Image;                
                monoInputItem.GradientMatrix = generatedGradientMatrix;
                MonoInput.MonoInputItems[inputIndex] = monoInputItem;

                var sum = TensorPrimitives.Sum(winnerMiniColumn.Temp_Hash);
                if (sum >= Constants.MinBitsInHashForMemory)
                {
                    var g = winnerMiniColumn.GetPictureAverageGradientInPoint();

                    int generatedMemoriesCount = 1;// random.Next(10) + 3;

                    foreach (var _ in Enumerable.Range(0, generatedMemoriesCount))
                    {
                        winnerMiniColumn.AddMemory(new Memory
                        {
                            Hash = (float[])winnerMiniColumn.Temp_Hash.Clone(),
                            PictureAverageGradientInPoint = g.Item3,
                            PictureInputIndex = inputIndex
                        });
                    }
                }
                else
                {
                    //throw new InvalidOperationException();
                }
            }
        }        

        public void DoStep_Memory(                        
            Random random
            )
        {
            ActivitiyMaxInfo activitiyMaxInfo = new();

            CurrentInputIndex += 1;

            var gradientMatrix = MonoInput.MonoInputItems[CurrentInputIndex].GradientMatrix;

            MiniColumn? winnerMiniColumn;

            for (; ; )
            {
                var mci = random.Next(Cortex.SubArea_MiniColumns.Length);
                MiniColumn mc = Cortex.SubArea_MiniColumns[mci];

                if (mc.Memories.Count == 0)
                    break;

                var mi = random.Next(mc.Memories.Count);

                Memory? memory = mc.Memories[mi];
                if (memory is null)
                    break;
                
                mc.Memories[mi] = null;

                CalculateDetectorsAndActivityAndSuperActivity(MonoInput.MonoInputItems[memory.PictureInputIndex].GradientMatrix, activitiyMaxInfo);

                // Сохраняем воспоминание в миниколонке-победителе.
                winnerMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
                if (winnerMiniColumn is not null)
                {
                    if (!ReferenceEquals(winnerMiniColumn, mc) &&
                        TensorPrimitives.Sum(winnerMiniColumn.Temp_Hash) >= Constants.MinBitsInHashForMemory)
                    {
                        var g = winnerMiniColumn.GetPictureAverageGradientInPoint();
                        winnerMiniColumn.AddMemory(new Memory
                        {
                            Hash = (float[])winnerMiniColumn.Temp_Hash.Clone(),
                            PictureAverageGradientInPoint = g.Item3,
                            PictureInputIndex = memory.PictureInputIndex
                        });
                    }
                    else
                    {                        
                        mc.Memories[mi] = memory;
                    }
                }

                break;
            }

            foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
            {
                MiniColumn mc = Cortex.SubArea_MiniColumns[mci];                

                foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                {
                    Memory? memory = mc.Memories[mi];
                    if (memory is null)
                        continue;

                    mc.Temp_Memories.Add(memory);
                }

                mc.Memories.Clear();
                mc.Memories.AddRange(mc.Temp_Memories);
                mc.Temp_Memories.Clear();
            }
        }

        public async Task ReorderMemoriesAsync(int iterationsCount, Random random, Func<Task>? refreshAction = null)
        {
            ActivitiyMaxInfo activitiyMaxInfo = new();

            Stopwatch sw = new();
            for (int iterationN = 0; iterationN < iterationsCount; iterationN += 1)
            {
                sw.Restart();

                int changedCount = 0;
                foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
                {
                    MiniColumn mc = Cortex.SubArea_MiniColumns[mci];

                    foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                    {
                        Memory? memory = mc.Memories[mi];
                        if (memory is null)
                            continue;
                        
                        mc.Memories[mi] = null;

                        CalculateDetectorsAndActivityAndSuperActivity(MonoInput.MonoInputItems[memory.PictureInputIndex].GradientMatrix, activitiyMaxInfo);

                        // Сохраняем воспоминание в миниколонке-победителе.
                        MiniColumn? winnerMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
                        if (winnerMiniColumn is not null)
                        {
                            if (!ReferenceEquals(winnerMiniColumn, mc) &&
                                TensorPrimitives.Sum(winnerMiniColumn.Temp_Hash) >= Constants.MinBitsInHashForMemory)
                            {
                                var g = winnerMiniColumn.GetPictureAverageGradientInPoint();
                                winnerMiniColumn.AddMemory(new Memory
                                {
                                    Hash = (float[])winnerMiniColumn.Temp_Hash.Clone(),
                                    PictureAverageGradientInPoint = g.Item3,
                                    PictureInputIndex = memory.PictureInputIndex
                                });
                                changedCount += 1;
                            }
                            else
                            {                                
                                mc.Memories[mi] = memory;
                            }
                        }

                        //Memory memory = mc.Memories[mi];
                        //if (memory.IsDeleted)
                        //    continue;

                        //memory.IsDeleted = true;
                        //mc.Memories[mi] = memory;

                        //GetSuperActivitiyMaxInfo(memory.Hash, activitiyMaxInfo);

                        //// Сохраняем воспоминание в миниколонке-победителе.
                        //winnerMiniColumn = activitiyMaxInfo.SuperActivityMax_MiniColumn;
                        //if (winnerMiniColumn is not null)
                        //{
                        //    memory.IsDeleted = false;
                        //    if (ReferenceEquals(winnerMiniColumn, mc))
                        //        mc.Memories[mi] = memory;
                        //    else
                        //        winnerMiniColumn.Memories.Add(memory);
                        //}
                    }
                }

                foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
                {
                    MiniColumn mc = Cortex.SubArea_MiniColumns[mci];                    

                    foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                    {
                        Memory? memory = mc.Memories[mi];
                        if (memory is null)
                            continue;

                        mc.Temp_Memories.Add(memory);
                    }

                    mc.Memories.Clear();
                    mc.Memories.AddRange(mc.Temp_Memories);
                    mc.Temp_Memories.Clear();
                }

                sw.Stop();

                UserFriendlyLogger.LogInformation($"ReorderMemories() iteration finished. ChangedCount: {changedCount}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");

                if (changedCount < 10)
                {   
                    break;
                }
                else
                {
                    if (refreshAction is not null)
                        await refreshAction();
                }
            }

            UserFriendlyLogger.LogInformation("ReorderMemories() finished.");
        }

        public void CalculateDetectorsAndActivityAndSuperActivity(DenseMatrix<GradientInPoint> gradientMatrix, ActivitiyMaxInfo activitiyMaxInfo)
        {
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(Retina, gradientMatrix, Constants);
                    });

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.GetHash(mc.Temp_Hash);
                    mc.Temp_Activity = MiniColumnsActivity.GetActivity(mc, mc.Temp_Hash, Constants);

#if CALC_BITS_COUNT_IN_HASH_HISTOGRAM
                    int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);
                    //dataToDisplayHolder.MiniColumsActivatedDetectorsCountDistribution[activatedDetectors.Intersect(miniColumn.Detectors).Count()] += 1;
                    DataToDisplayHolder.MiniColumsBitsCountInHashDistribution[bitsCountInHash] += 1;
#endif
                });

            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

            if (Constants.SuperactivityThreshold)
                activitiyMaxInfo.MaxSuperActivity = Constants.K2 - Constants.K0 + 0.01f; // Чуть больше, чем активность пустой миниколонки
            else
                activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

            foreach (var mc in Cortex.SubArea_MiniColumns)
            {
                mc.Temp_SuperActivity = MiniColumnsActivity.GetSuperActivity(mc, Constants);

                float a = mc.Temp_Activity.Item3;
                if (a > activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.MaxActivity = a;
                    activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
                }
                else if (a == activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
                }

                if (mc.Temp_SuperActivity > activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
                }
                else if (mc.Temp_SuperActivity == activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
                }
            }
        }

        //public void CalculateDetectorsAndActivityAndSuperActivity_Simplified(DenseMatrix<GradientInPoint> gradientMatrix, ActivitiyMaxInfo activitiyMaxInfo)
        //{
        //    Parallel.For(
        //            fromInclusive: 0,
        //            toExclusive: Cortex.SubArea_Detectors.Length,
        //            di =>
        //            {
        //                var d = Cortex.SubArea_Detectors[di];
        //                d.CalculateIsActivated(gradientMatrix, Constants);
        //            });

        //    var centerMiniColumn_Temp_Hash = Cortex.CenterMiniColumn!.Temp_Hash;
        //    Cortex.CenterMiniColumn!.GetHash(centerMiniColumn_Temp_Hash);
        //    Parallel.For(
        //        fromInclusive: 0,
        //        toExclusive: Cortex.SubArea_MiniColumns.Length,
        //        mci =>
        //        {
        //            var mc = Cortex.SubArea_MiniColumns[mci];
        //            Array.Copy(centerMiniColumn_Temp_Hash, mc.Temp_Hash, centerMiniColumn_Temp_Hash.Length);
        //            mc.Temp_Activity = MiniColumnsActivity.GetActivity(mc, mc.Temp_Hash, Cortex);
        //        });

        //    activitiyMaxInfo.MaxActivity = float.MinValue;
        //    activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

        //    activitiyMaxInfo.MaxSuperActivity = float.MinValue;
        //    activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

        //    foreach (var mc in Cortex.SubArea_MiniColumns)
        //    {
        //        mc.Temp_SuperActivity = MiniColumnsActivity.GetSuperActivity(mc, Cortex);

        //        float a = mc.Temp_Activity.Item3;
        //        if (a > activitiyMaxInfo.MaxActivity)
        //        {
        //            activitiyMaxInfo.MaxActivity = a;
        //            activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
        //            activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
        //        }
        //        else if (a == activitiyMaxInfo.MaxActivity)
        //        {
        //            activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
        //        }

        //        if (mc.Temp_SuperActivity > activitiyMaxInfo.MaxSuperActivity)
        //        {
        //            activitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
        //            activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
        //            activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
        //        }
        //        else if (mc.Temp_SuperActivity == activitiyMaxInfo.MaxSuperActivity)
        //        {
        //            activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
        //        }
        //    }
        //}

        #endregion

        private async Task DoStepAsync(
            int inputIndex, 
            DenseMatrix<GradientInPoint> gradientMatrix, 
            ActivitiyMaxInfo activitiyMaxInfo, 
            Random random, 
            bool randomInitialization,
            bool reorderMemoriesPeriodically)
        {
            // Sleep and refresh all minicolumns
            if (reorderMemoriesPeriodically && inputIndex > 0 && inputIndex % 2000 == 0)
            {
                await ReorderMemoriesAsync(1, random);
            }

            CalculateDetectorsAndActivityAndSuperActivity(gradientMatrix, activitiyMaxInfo);

            MiniColumn? winnerMiniColumn;
            // Сохраняем воспоминание в миниколонке-победителе.
            if (randomInitialization)
            {
                var winnerIndex = random.Next(Cortex.SubArea_MiniColumns.Length);                
                winnerMiniColumn = Cortex.SubArea_MiniColumns[winnerIndex];
            }
            else
            {
                winnerMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
            }            
            Cortex.Temp_SuperActivityMax_MiniColumn = winnerMiniColumn;
            if (winnerMiniColumn is not null)
            {       
                if (TensorPrimitives.Sum(winnerMiniColumn.Temp_Hash) >= Constants.MinBitsInHashForMemory)
                {
                    var g = winnerMiniColumn.GetPictureAverageGradientInPoint();
                    winnerMiniColumn.AddMemory(new Memory
                    {
                        Hash = (float[])winnerMiniColumn.Temp_Hash.Clone(),
                        PictureAverageGradientInPoint = g.Item3,
                        PictureInputIndex = inputIndex
                    });                    
                    //Cortex.Temp_WinnerMiniColumn_AverageGradientInPoint_Delta = Math.Sqrt((g.Item1.GradX - g.Item2.GradX) * (g.Item1.GradX - g.Item2.GradX) +
                    //    (g.Item1.GradY - g.Item2.GradY) * (g.Item1.GradY - g.Item2.GradY));
                    //if (Cortex.Temp_WinnerMiniColumn_AverageGradientInPoint_Delta < 1000)
                    //if (g.Item1.GradX * g.Item2.GradX >= 0 && g.Item1.GradY * g.Item2.GradY >= 0)
                    //{
                    //    winnerMiniColumn.Memories.Add(new Memory
                    //    {
                    //        Hash = (float[])winnerMiniColumn.Temp_Hash.Clone(),
                    //        AverageGradientInPoint = g.Item3,
                    //        InputIndex = inputIndex
                    //    });
                    //}
                    //else
                    //{
                    //}

                    Cortex.Temp_WinnerMiniColumn_AverageGradientInPoint_Magnitude = g.Item3.Magnitude;
                }   
                else
                {
                    Cortex.Temp_WinnerMiniColumn_AverageGradientInPoint_Magnitude = Double.NaN;
                    Cortex.Temp_WinnerMiniColumn_AverageGradientInPoint_Delta = Double.NaN;
                }
            }
        }        

        private (DenseMatrix<GradientInPoint>, Bitmap) GetGeneratedLine_GradientMatrix(double positionK, double angle)
        {
            // Создаем изображение размером 280x280           

            Generated_CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0);
            Generated_CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + Generated_CenterXDelta;
            Generated_CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            if (angle < 0)
                angle += 2 * Math.PI;
            Generated_AngleDelta = angle;
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

            int smallWidth = MNISTHelper.MNISTImageWidthPixels;
            int smallHeight = MNISTHelper.MNISTImageHeightPixels;

            Bitmap resizedBitmap = new Bitmap(smallWidth, smallHeight);
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
                g.DrawImage(originalBitmap, new Rectangle(0, 0, smallWidth, smallHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению            
            return (SobelOperator.ApplySobel(resizedBitmap, smallWidth, smallHeight), resizedBitmap);
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
        public class ModelConstants : IConstants
        {
            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidthPixels => MNISTHelper.MNISTImageWidthPixels;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeightPixels => MNISTHelper.MNISTImageHeightPixels;

            /// <summary>
            ///     Не используется
            /// </summary>
            public int AngleRangeDegree_LimitMagnitude => 70;// Sigmoid = 70 for pi/2; Linear = 150

            public double DetectorMinGradientMagnitude => 5;

            public int GeneratedMinGradientMagnitude => 5;

            public int GeneratedMaxGradientMagnitude => 1200;

            public int AngleRangeDegreeMin => 120;

            public int AngleRangeDegreeMax => 360;

            public int MagnitudeRangesCount => 4;

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
            ///     Расстояние между детекторами по горизонтали и вертикали 
            ///     [0..MNISTImageWidth]
            /// </summary>
            public double DetectorDelta => 0.05;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 600;  // ORIG 250         

            public int HashLength => 300;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 400; //400;

            /// <summary>
            ///     Индекс X центра подобласти [0..CortexWidth]
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти [0..CortexHeight]
            /// </summary>
            public int SubAreaCenter_Cy => 100;                       

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 5; // 11 optimum

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int MiniColumnsMaxDistance => 3;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Длина короткого хэш-вектора
            /// </summary>
            public int ShortHashLength => 50;

            /// <summary>
            ///     Количество бит в коротком хэш-векторе
            /// </summary>
            public int ShortHashBitsCount => 11;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public float MemoryClustersThreshold => 0.66f;

            public int Angle_SmallPoints_Count => 1000;

            public float Angle_SmallPoints_Radius => 0.003f;

            public int Angle_BigPoints_Count => 200;

            public float Angle_BigPoints_Radius => 0.015f;

            /// <summary>
            ///     Нулевой уровень косинусного расстояния
            /// </summary>
            public float K0 { get; set; }
            /// <summary>
            ///     Порог косинусного расстояния для учета 
            /// </summary>
            public float K1 { get; set; }
            /// <summary>
            ///     Косинусное расстояние для пустой колонки
            /// </summary>
            public float K2 { get; set; }

            /// <summary>
            ///     K значимости соседей
            /// </summary>
            public float K3 { get; set; }

            public float K4 { get; set; }

            public float K5 { get; set; }

            public bool SuperactivityThreshold { get; set; }
        }        
    }
}


///// <summary>
/////     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
///// </summary>
//public int InitialMemoryBitsCount => 11;

//public Image[] GetImages3()
//{
//    Random random = new();
//    //var hash0 = new float[Constants.HashLength];
//    //foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
//    //{
//    //    hash0[random.Next(hash0.Length)] = 1.0f;
//    //}            

//    Cortex.VisualizationTableItems.Clear();

//    //int currentMnistImageIndex = 0;
//    var centerMiniColumn_Hash = new float[Constants.HashLength];
//    foreach (var currentMnistImageIndex in Enumerable.Range(500, 2000))
//    {
//        //currentMnistImageIndex += 1;

//        var gradientMatrix = MonoInput.MonoInputItems[CurrentInputIndex].GradientMatrix;

//        Parallel.For(
//            fromInclusive: 0,
//            toExclusive: Cortex.SubArea_Detectors.Length,
//            di =>
//            {
//                var d = Cortex.SubArea_Detectors[di];
//                d.CalculateIsActivated(gradientMatrix);
//            });

//        var centerMiniColumn = Cortex.CenterMiniColumn!;
//        centerMiniColumn.GetHash(centerMiniColumn_Hash);

//        if (TensorPrimitives.Sum(centerMiniColumn_Hash) < Constants.MinBitsInHashForMemory)
//            continue;

//        bool found = false;

//        Parallel.For(
//            fromInclusive: 0,
//            toExclusive: Cortex.VisualizationTableItems.Count,
//            (di, pls) =>
//            {
//                var visualizationTableItem = Cortex.VisualizationTableItems[di];
//                var cosineSimilarity = TensorPrimitives.CosineSimilarity(centerMiniColumn_Hash, visualizationTableItem.Hash);
//                if (cosineSimilarity > 0.90)
//                {
//                    found = true;
//                    pls.Stop();
//                }
//            });

//        if (!found)
//        {
//            //var bitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

//            //var subBitmap = BitmapHelper.GetSubBitmap(
//            //    bitmap, 
//            //    (int)(centerMiniColumn.CenterX / Constants.DetectorDelta),
//            //    (int)(centerMiniColumn.CenterY / Constants.DetectorDelta),
//            //    Cortex.DetectorsVisibleRadius + 2);
//            // //Bitmap image = MNISTHelper.GetBitmap(Images[CurrentInputIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight, );

//            Color color = Color.Black;

//            VisualizationTableItem visualizationTableItem = new()
//            {
//                Hash = (float[])centerMiniColumn_Hash.Clone(),
//                Color = color,
//                //Image = subBitmap
//                SubArea_MiniColumns_Hashes = new float[Cortex.SubArea_MiniColumns.Length][]
//            };
//            foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
//            {
//                float[] hash = new float[Constants.HashLength];
//                MiniColumn mc = Cortex.SubArea_MiniColumns[mci];
//                mc.GetHash(hash);
//                visualizationTableItem.SubArea_MiniColumns_Hashes[mci] = hash;
//            }
//            Cortex.VisualizationTableItems.Add(visualizationTableItem);
//        }
//    }

//    SetColors_VisualizationTableItems(Cortex);

//    foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
//    {
//        MiniColumn mc = Cortex.SubArea_MiniColumns[mci];
//        mc.Temp_ActivityColor = Color.Black;
//        mc.Temp_SuperActivityColor = Color.Black;
//    }

//    ActivitiyMaxInfo activitiyMaxInfo = new();
//    foreach (var vti in Enumerable.Range(0, Cortex.VisualizationTableItems.Count))
//    {
//        var visualizationTableItem = Cortex.VisualizationTableItems[vti];

//        GetSuperActivitiyMaxInfo2(visualizationTableItem, activitiyMaxInfo);

//        MiniColumn? winnerMiniColumn = activitiyMaxInfo.GetActivityMax_MiniColumn(random);
//        if (winnerMiniColumn is not null)
//        {
//            winnerMiniColumn.Temp_ActivityColor = visualizationTableItem.Color;
//        }

//        winnerMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
//        if (winnerMiniColumn is not null)
//        {
//            winnerMiniColumn.Temp_SuperActivityColor = visualizationTableItem.Color;
//        }
//    }

//    var image0 = BitmapHelper.GetSubBitmap(
//        Visualisation.GetBitmapFromMiniColums_ActivityColor(Cortex),
//        Cortex.MiniColumns.Dimensions[0] / 2,
//        Cortex.MiniColumns.Dimensions[1] / 2,
//        Cortex.SubAreaMiniColumnsRadius + 2);

//    var image1 = BitmapHelper.GetSubBitmap(
//        Visualisation.GetBitmapFromMiniColums_SuperActivityColor(Cortex),
//        Cortex.MiniColumns.Dimensions[0] / 2,
//        Cortex.MiniColumns.Dimensions[1] / 2,
//        Cortex.SubAreaMiniColumnsRadius + 2);

//    var image2 = BitmapHelper.GetSubBitmap(
//        Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex),
//        Cortex.MiniColumns.Dimensions[0] / 2,
//        Cortex.MiniColumns.Dimensions[1] / 2,
//        Cortex.SubAreaMiniColumnsRadius + 2);

//    return [image0, image1, image2];
//}


//private void SetColors_VisualizationTableItems(Cortex cortex)
//{
//    Random random = new();

//    VisualizationTableItemsCluster r_VisualizationTableItemsCluster = new()
//    {
//        Hash = new float[Constants.HashLength],
//        TempHash = new float[Constants.HashLength],
//        VisualizationTableItems = new(1000),
//        MinCosineSimilarity = float.MaxValue,
//        MaxCosineSimilarity = float.MinValue,
//    };
//    VisualizationTableItemsCluster g_VisualizationTableItemsCluster = new()
//    {
//        Hash = new float[Constants.HashLength],
//        TempHash = new float[Constants.HashLength],
//        VisualizationTableItems = new(1000),
//        MinCosineSimilarity = float.MaxValue,
//        MaxCosineSimilarity = float.MinValue,
//    };
//    VisualizationTableItemsCluster b_VisualizationTableItemsCluster = new()
//    {
//        Hash = new float[Constants.HashLength],
//        TempHash = new float[Constants.HashLength],
//        VisualizationTableItems = new(1000),
//        MinCosineSimilarity = float.MaxValue,
//        MaxCosineSimilarity = float.MinValue,
//    };
//    VisualizationTableItemsCluster[] clusters = [r_VisualizationTableItemsCluster, g_VisualizationTableItemsCluster, b_VisualizationTableItemsCluster];

//    // Случайные начальные центры
//    foreach (var cluster in clusters)
//    {
//        foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
//        {
//            cluster.Hash[random.Next(Constants.HashLength)] = 1.0f;
//        }
//    }

//    // Находим центры кластеров (EM)
//    foreach (var _ in Enumerable.Range(0, 5))
//    {
//        foreach (var cluster in clusters)
//        {
//            cluster.VisualizationTableItems.Clear();
//        }

//        foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
//        {
//            var max = clusters.MaxBy(c => TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, c.Hash));
//            if (max is null)
//            {
//                throw new Exception();
//            }
//            else
//            {
//                max.VisualizationTableItems.Add(visualizationTableItem);
//            }
//        }

//        foreach (var cluster in clusters)
//        {
//            Array.Clear(cluster.Hash);
//            if (cluster.VisualizationTableItems.Count == 0)
//                throw new Exception();
//            foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
//            {
//                TensorPrimitives.Add(cluster.Hash, visualizationTableItem.Hash, cluster.Hash);
//            }
//            TensorPrimitives.Divide(cluster.Hash, cluster.VisualizationTableItems.Count, cluster.Hash);
//        }
//    }

//    //// Находим максимально удаленные точки от цетров кластеров
//    //foreach (var cluster in clusters)
//    //{
//    //    var otherClusters = clusters.Where(c => !ReferenceEquals(c, cluster)).ToArray();
//    //    float min = float.MaxValue;
//    //    VisualizationTableItem? minVisualizationTableItem = null;
//    //    foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
//    //    {
//    //        float d = 0.0f;
//    //        foreach (var otherCluster in otherClusters)
//    //        {
//    //            d += TensorPrimitives.CosineSimilarity(otherCluster.Hash, visualizationTableItem.Hash);
//    //        }
//    //        if (d < min)
//    //            minVisualizationTableItem = visualizationTableItem;
//    //    }
//    //    cluster.TempHash = minVisualizationTableItem!.Hash;
//    //}

//    //foreach (var cluster in clusters)
//    //{
//    //    Array.Copy(cluster.TempHash, cluster.Hash, cluster.TempHash.Length);
//    //}

//    foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
//    {
//        foreach (var cluster in clusters)
//        {
//            var d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, cluster.Hash);
//            if (d < cluster.MinCosineSimilarity)
//                cluster.MinCosineSimilarity = d;
//            if (d > cluster.MaxCosineSimilarity)
//                cluster.MaxCosineSimilarity = d;
//        }
//    }
//    foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
//    {
//        var r_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, r_VisualizationTableItemsCluster.Hash);
//        int r = 1 + (int)(254 * (r_d - r_VisualizationTableItemsCluster.MinCosineSimilarity) /
//            (r_VisualizationTableItemsCluster.MaxCosineSimilarity - r_VisualizationTableItemsCluster.MinCosineSimilarity));
//        var g_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, g_VisualizationTableItemsCluster.Hash);
//        int g = 1 + (int)(254 * (g_d - g_VisualizationTableItemsCluster.MinCosineSimilarity) /
//            (g_VisualizationTableItemsCluster.MaxCosineSimilarity - g_VisualizationTableItemsCluster.MinCosineSimilarity));
//        var b_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, b_VisualizationTableItemsCluster.Hash);
//        int b = 1 + (int)(254 * (b_d - b_VisualizationTableItemsCluster.MinCosineSimilarity) /
//            (b_VisualizationTableItemsCluster.MaxCosineSimilarity - b_VisualizationTableItemsCluster.MinCosineSimilarity));

//        float k = 255.0f / Math.Max(r, Math.Max(g, b));
//        if (float.IsInfinity(k) || float.IsNaN(k))
//            throw new Exception();
//        if (r == 1 && g == 1 && b == 1)
//            throw new Exception();
//        visualizationTableItem.Color = Color.FromArgb((int)(k * r), (int)(k * g), (int)(k * b));
//    }
//}

//private void SetColors_VisualizationTableItems2(Cortex cortex)
//{
//    Random random = new();

//    VisualizationTableItemsCluster[] clusters = new VisualizationTableItemsCluster[7];

//    foreach (var ci in Enumerable.Range(0, clusters.Length))
//    {
//        clusters[ci] = new()
//        {
//            Hash = new float[Constants.HashLength],
//            TempHash = new float[Constants.HashLength],
//            VisualizationTableItems = new(1000),
//            MinCosineSimilarity = float.MaxValue,
//            MaxCosineSimilarity = float.MinValue,
//        };
//    }

//    // Случайные начальные центры
//    foreach (var cluster in clusters)
//    {
//        foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
//        {
//            cluster.Hash[random.Next(Constants.HashLength)] = 1.0f;
//        }
//    }

//    // Находим центры кластеров (EM)
//    foreach (var _ in Enumerable.Range(0, 5))
//    {
//        foreach (var cluster in clusters)
//        {
//            cluster.VisualizationTableItems.Clear();
//        }

//        foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
//        {
//            var max = clusters.MaxBy(c => TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, c.Hash));
//            if (max is null)
//            {
//                throw new Exception();
//            }
//            else
//            {
//                max.VisualizationTableItems.Add(visualizationTableItem);
//            }
//        }

//        foreach (var cluster in clusters)
//        {
//            Array.Clear(cluster.Hash);
//            if (cluster.VisualizationTableItems.Count == 0)
//                throw new Exception();
//            foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
//            {
//                TensorPrimitives.Add(cluster.Hash, visualizationTableItem.Hash, cluster.Hash);
//            }
//            TensorPrimitives.Divide(cluster.Hash, cluster.VisualizationTableItems.Count, cluster.Hash);
//        }
//    }

//    foreach (var ci in Enumerable.Range(0, clusters.Length))
//    {
//        var cluster = clusters[ci];
//        foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
//        {
//            visualizationTableItem.Color = DefaultColors[ci];
//        }
//    }
//}


//private void GetSuperActivitiyMaxInfo(float[] hash, ActivitiyMaxInfo activitiyMaxInfo)
//{
//    Parallel.For(
//        fromInclusive: 0,
//        toExclusive: Cortex.SubArea_MiniColumns.Length,
//        mci =>
//        {
//            var mc = Cortex.SubArea_MiniColumns[mci];
//            mc.Temp_Activity = MiniColumnsActivity.GetActivity(mc, hash, Cortex);
//        });

//    GetSuperActivitiyMaxInfo(activitiyMaxInfo);
//}

//private void GetSuperActivitiyMaxInfo2(VisualizationTableItem visualizationTableItem, ActivitiyMaxInfo activitiyMaxInfo)
//{
//    Parallel.For(
//        fromInclusive: 0,
//        toExclusive: Cortex.SubArea_MiniColumns.Length,
//        mci =>
//        {
//            var mc = Cortex.SubArea_MiniColumns[mci];
//            mc.Temp_Activity = MiniColumnsActivity.GetActivity(mc, visualizationTableItem.SubArea_MiniColumns_Hashes[mci], Cortex);
//        });

//    GetSuperActivitiyMaxInfo(activitiyMaxInfo);
//}


//private void GetSuperActivitiyMaxInfo(ActivitiyMaxInfo activitiyMaxInfo)
//{


//    //Parallel.For(
//    //        fromInclusive: 0,
//    //        toExclusive: Cortex.SubArea_MiniColumns.Length,
//    //        () => new ActivitiyMaxInfo(), // method to initialize the local variable
//    //        (mci, loopState, localActivitiyMaxInfo) => // method invoked by the loop on each iteration
//    //        {
//    //            var mc = Cortex.SubArea_MiniColumns[mci];
//    //            mc.Temp_SuperActivity = mc.GetSuperActivity();

//    //            float a = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
//    //            if (a > localActivitiyMaxInfo.MaxActivity)
//    //            {
//    //                localActivitiyMaxInfo.MaxActivity = a;
//    //                localActivitiyMaxInfo.ActivityMax_MiniColumn = mc;
//    //            }

//    //            if (mc.Temp_SuperActivity > localActivitiyMaxInfo.MaxSuperActivity)
//    //            {
//    //                localActivitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
//    //                localActivitiyMaxInfo.SuperActivityMax_MiniColumn = mc;
//    //            }

//    //            return localActivitiyMaxInfo; // value to be passed to next iteration
//    //        },
//    //        localActivitiyMaxInfo => // Method to be executed when each partition has completed.
//    //        {
//    //            lock (activitiyMaxInfo)
//    //            {
//    //                if (localActivitiyMaxInfo.MaxActivity > activitiyMaxInfo.MaxActivity)
//    //                {
//    //                    activitiyMaxInfo.MaxActivity = localActivitiyMaxInfo.MaxActivity;
//    //                    activitiyMaxInfo.ActivityMax_MiniColumn = localActivitiyMaxInfo.ActivityMax_MiniColumn;
//    //                }

//    //                if (localActivitiyMaxInfo.MaxSuperActivity > activitiyMaxInfo.MaxSuperActivity)
//    //                {
//    //                    activitiyMaxInfo.MaxSuperActivity = localActivitiyMaxInfo.MaxSuperActivity;
//    //                    activitiyMaxInfo.SuperActivityMax_MiniColumn = localActivitiyMaxInfo.SuperActivityMax_MiniColumn;
//    //                }
//    //            }
//    //        });
//}        

//private Image GetMnistTotalBitmap()
//{
//    GradientInPoint[,] totalGradientMatrix = new GradientInPoint[MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels];
//    foreach (int i in Enumerable.Range(0, MonoInput.MonoInputItems.Length))
//    {
//        DenseMatrix<GradientInPoint> gm = MonoInput.MonoInputItems[i].GradientMatrix;
//        for (int y = 1; y < MNISTHelper.MNISTImageHeightPixels - 1; y += 1)
//        {
//            for (int x = 1; x < MNISTHelper.MNISTImageWidthPixels - 1; x += 1)
//            {
//                GradientInPoint p = gm[x, y];
//                GradientInPoint totalP = totalGradientMatrix[x, y];

//                totalP.GradX += p.GradX;
//                totalP.GradY += p.GradY;                        

//                totalGradientMatrix[x, y] = totalP;
//            }
//        }                
//    }
//    //for (int y = 1; y < MNISTHelper.MNISTImageHeight - 1; y += 1)
//    //{
//    //    for (int x = 1; x < MNISTHelper.MNISTImageWidth - 1; x += 1)
//    //    {                    
//    //        GradientInPoint totalP = totalGradientMatrix[x, y];

//    //        //totalP.GradX = totalP.GradX / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения
//    //        //totalP.GradY = totalP.GradY / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения
//    //        //totalP.Magnitude = totalP.Magnitude / GradientMatricesCollection.Count;
//    //        //totalP.Angle = totalP.Angle / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения

//    //        totalGradientMatrix[x, y] = totalP;
//    //    }
//    //}

//    return Visualisation.GetGradientBigBitmapObsolete(totalGradientMatrix);
//}   