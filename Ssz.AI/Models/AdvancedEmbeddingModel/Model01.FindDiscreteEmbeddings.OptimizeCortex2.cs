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
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model01
    {
        public const int TopProxPointsCount2 = 60;
        public const int TopProxPrimaryPointsCount2 = 300;

        public void OptimizeCortex2(LanguageInfo languageInfo, ILoggersSet loggersSet, Clusterization_AlgorithmData clusterization_AlgorithmData)
        { 
            var totalStopwatch = Stopwatch.StartNew();
            var stopwatch = new Stopwatch();
            var r = new Random();

            #region Initialization

            Cortex = new(xCount: (int)Math.Sqrt(languageInfo.Words.Count) + 1, yCount: (int)Math.Sqrt(languageInfo.Words.Count) + 1);
            //if (Words.Count > Cortex.Array.Length)
            //{
            //    loggersSet.UserFriendlyLogger.LogError("Cortex size too low; Words.Count = " + Words.Count);
            //    return;
            //}

            int ix, iy;

            #region Random Points positions in Cortex  
            
            for (int wordIndex = 0; wordIndex < languageInfo.Words.Count; wordIndex += 1)
            {
                for (; ; )
                {
                    ix = r.Next(Cortex.XCount);
                    iy = r.Next(Cortex.YCount);
                    ref var pointRef = ref Cortex[ix, iy];
                    if (pointRef is null)
                    {
                        Point point = new()
                        {
                            WordIndex = wordIndex,
                            V = [ix, iy]
                        };
                        pointRef = point;
                        languageInfo.Words[wordIndex].Point = point;
                        break;
                    }
                }
            }
            for (ix = 0; ix < Cortex.XCount; ix += 1)
            {
                for (iy = 0; iy < Cortex.YCount; iy += 1)
                {
                    ref var pointRef = ref Cortex[ix, iy];
                    if (pointRef is null)
                    {
                        Point emptyPoint = new()
                        {
                            WordIndex = -1,
                            V = [ix, iy]
                        };
                        pointRef = emptyPoint;
                        languageInfo.Words[0].Point = emptyPoint; // Ref to any empty point.
                    }
                }
            }

            loggersSet.UserFriendlyLogger.LogInformation("Random initialization done.");

            #endregion

            #region point.TopProxPoints calculation

            stopwatch.Restart();

            for (int wordIndex = 0; wordIndex < languageInfo.Words.Count; wordIndex += 1)
            {
                languageInfo.Words[wordIndex].Temp_Flag = false;
            }
            // TODO
            //for (int i = 0; i < clusterization_AlgorithmData.PrimaryWords!.Length; i += 1)
            //{
            //    Word primaryWord = clusterization_AlgorithmData.PrimaryWords[i];
            //    primaryWord.Temp_Flag = true; // PrimaryWord
            //}

            for (ix = 0; ix < Cortex.XCount; ix += 1)
            {
                for (iy = 0; iy < Cortex.YCount; iy += 1)
                {
                    Point point = Cortex[ix, iy]!;
                    if (point.WordIndex == -1)
                    {
                        point.Temp_TopProxPoints = new (float, Point)[0];
                    }
                    else
                    {   
                        point.Temp_TopProxPoints = Cortex.Array
                            .Where(point2 => point2.WordIndex != -1 && point2.WordIndex != point.WordIndex)
                            .Select(point2 => (languageInfo.WordsDistancesOldMatrix[point.WordIndex, point2.WordIndex], point2))
                            .OrderBy(i => i.Item1)
                            .Take(TopProxPointsCount2)
                            .Where(it => it.Item1 > 0.0)
                            .ToArray();
                        if (languageInfo.Words[point.WordIndex].Temp_Flag) // Primary word
                        {
                            point.Temp_TopProxPrimaryPoints = Cortex.Array
                                .Where(point2 => point2.WordIndex != -1 && point2.WordIndex != point.WordIndex && languageInfo.Words[point2.WordIndex].Temp_Flag)
                                .Select(point2 => (languageInfo.WordsDistancesOldMatrix[point.WordIndex, point2.WordIndex], point2))
                                .OrderBy(i => i.Item1)
                                .Take(TopProxPrimaryPointsCount2)
                                .ToArray();
                            point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.PrimaryPoint;
                        }
                    }
                }
            }            

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("TopProxPoints Calculation done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            #endregion

            #region point.GroupId initialization

            for (int groupId = 1; groupId <= 5; groupId += 1)
            {
                ix = r.Next(Cortex.XCount);
                iy = r.Next(Cortex.YCount);
                Point point = Cortex[ix, iy];
                point.GroupId_ToDisplay = groupId;
                foreach (var it in point.Temp_TopProxPoints!)
                {
                    it.Item2.GroupId_ToDisplay = groupId;
                }
            }

            #endregion

            #region CortexCopy Initialization

            lock (CortexCopySyncRoot)
            {
                CortexCopy = new Cortex(Cortex.XCount, Cortex.YCount);
                for (int i = 0; i < Cortex.Array.Length; i += 1)
                {
                    Point point = new()
                    {
                        V = new float[2]
                    };
                    point.CopyData(Cortex.Array[i]);
                    CortexCopy.Array[i] = point;
                }
            }

            #endregion

            #endregion

            int unchangedCount = 0;
            for (; ; )
            {
                stopwatch.Restart();

                bool changed = false;
                for (int n = 1; n <= Cortex.Array.Length; n += 1)
                {
                    if (OrgStep2(r.Next(Cortex.XCount), r.Next(Cortex.YCount)))
                        changed = true;
                }
                if (!changed)
                    unchangedCount += 1;
                else
                    unchangedCount = 0; // Reset counter

                CreateCortexCopy();

                stopwatch.Stop();
                loggersSet.UserFriendlyLogger.LogInformation($"Changed: {changed}; {Cortex.Array.Length} OrgSteps done. Elapsed Milliseconds {stopwatch.ElapsedMilliseconds}");

                if (unchangedCount >= 2)
                    break;
            }            

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("OrgStep totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }  

        public bool OrgStep2(int ix, int iy)
        {
            double pointEnergyOld;
            double point2EnergyOld;
            double pointsEnergySumNew;
            double maxDelta, delta;
            int ix2Result = 0;
            int iy2Result = 0;

            pointEnergyOld = GetPointEnergy2(ix, iy);            
            
            maxDelta = 0.0;

            for (var ix2 = ix - 1; ix2 <= ix + 1; ix2 += 1)
            {
                for (var iy2 = iy - 1; iy2 <= iy + 1; iy2 += 1)
                {
                    if (ix2 >= 0 && ix2 < Cortex.XCount && iy2 >= 0 && iy2 < Cortex.YCount && !(ix2 == ix && iy2 == iy))
                    {
                        point2EnergyOld = GetPointEnergy2(ix2, iy2);

                        // Exchange positions
                        SwapPoints2(ix, iy, ix2, iy2);

                        pointsEnergySumNew = GetPointEnergy2(ix2, iy2) + GetPointEnergy2(ix, iy);

                        // Return to old positions
                        SwapPoints2(ix, iy, ix2, iy2);

                        delta = pointEnergyOld + point2EnergyOld - pointsEnergySumNew;

                        if (delta > maxDelta)
                        {
                            maxDelta = delta;
                            ix2Result = ix2;
                            iy2Result = iy2;
                        }
                    }
                }
            }

            if (maxDelta > Double.Epsilon * 1000)
            {
                SwapPoints2(ix, iy, ix2Result, iy2Result);
                return true;
            }

            return false;
        }

        private void SwapPoints2(int ix1, int iy1, int ix2, int iy2)
        {
            ref var pointRef1 = ref Cortex[ix1, iy1];
            ref var pointRef2 = ref Cortex[ix2, iy2];

            Point? pointTemp = pointRef1;
            pointRef1 = pointRef2;
            pointRef2 = pointTemp;

            pointRef1.V[0] = ix1;
            pointRef1.V[1] = iy1;

            pointRef2.V[0] = ix2;
            pointRef2.V[1] = iy2;
        }

        /// <summary>
        ///     Энергия узла
        /// </summary>
        /// <param name="ix"></param>
        /// <param name="iy"></param>
        /// <returns></returns>
        private double GetPointEnergy2(int ix, int iy)
        {
            Point point = Cortex[ix, iy];            

            double pointEnergy = 0.0;

            float[] difference = new float[2];

            foreach (var it in point.Temp_TopProxPoints!)
            {
                pointEnergy += it.Item1 * it.Item1 * TensorPrimitives.Distance(point.V, it.Item2.V);
                //TensorPrimitives.Subtract(point.V, it.Item2.V, difference);
                //pointEnergy += it.Item1 * it.Item1 * TensorPrimitives.SumOfSquares(difference);                
            }

            if (point.Temp_TopProxPrimaryPoints is not null)
            {
                foreach (var it in point.Temp_TopProxPrimaryPoints)
                {
                    TensorPrimitives.Subtract(point.V, it.Item2.V, difference);
                    pointEnergy += 0.01f / TensorPrimitives.SumOfSquares(difference); 
                    //float distance = TensorPrimitives.Distance(point.V, it.Item2.V);
                    //pointEnergy += 100f / distance; // если 1000, то расползаются по стенкам
                }
            }            

            return pointEnergy;
        }

        private void CreateCortexCopy()
        {
            lock (CortexCopySyncRoot)
            {
                for (int i = 0; i < Cortex.Array.Length; i += 1)
                {
                    CortexCopy.Array[i].CopyData(Cortex.Array[i]);
                }
            }
        }

        private void CortexSaveToFile(ILoggersSet loggersSet)
        {
            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            using (MemoryStream memoryStream = new())
            using (SerializationWriter writer = new(memoryStream))
            {
                Cortex.SerializeOwnedData(writer, null);
                byte[] bytes = memoryStream.ToArray();
                File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, "Cortex.bin"), bytes);
            }
        }

        private void CortexLoadFromFile(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, "Cortex.bin"));
            using (SerializationReader reader = new(bytes))
            {
                Cortex = new Cortex();
                Cortex.DeserializeOwnedData(reader, null);
            }

            int ix, iy;
            for (ix = 0; ix < Cortex.XCount; ix += 1)
            {
                for (iy = 0; iy < Cortex.YCount; iy += 1)
                {
                    ref var pointRef = ref Cortex[ix, iy];
                    if (pointRef.WordIndex >= 0)
                        languageInfo.Words[pointRef.WordIndex].Point = pointRef;
                }
            }

            #region CortexCopy Initialization

            lock (CortexCopySyncRoot)
            {
                CortexCopy = new Cortex(Cortex.XCount, Cortex.YCount);
                for (int i = 0; i < Cortex.Array.Length; i += 1)
                {
                    Point point = new()
                    {
                        V = new float[2]
                    };
                    point.CopyData(Cortex.Array[i]);
                    CortexCopy.Array[i] = point;
                }
            }

            #endregion

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("CortexLoadFromFile done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }
    }

    public class Cortex : IOwnedDataSerializable
    {
        public Cortex()
        {
        }

        public Cortex(int xCount, int yCount)
        {
            XCount = xCount;
            YCount = yCount;
            Array = new Point[xCount * yCount];
        }

        public int XCount;

        public int YCount;

        public Point[] Array = null!;

        public ref Point this[int ix, int iy]
        {
            get { return ref Array[iy * XCount + ix]; }
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            writer.Write(XCount);
            writer.Write(YCount);
            writer.Write(Array.Length);
            for (int i = 0; i < Array.Length; i += 1)
            {
                var point = Array[i];
                writer.WriteOptimized(point.WordIndex);
                writer.WriteOptimized(point.GroupId_ToDisplay);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            XCount = reader.ReadInt32();
            YCount = reader.ReadInt32();
            int arrayLength = reader.ReadInt32();
            Array = new Point[arrayLength];
            int ix = 0;
            int iy = 0;
            for (int i = 0; i < arrayLength; i += 1)
            {
                Point point = new Point();
                Array[i] = point;
                point.WordIndex = reader.ReadOptimizedInt32();
                point.GroupId_ToDisplay = reader.ReadOptimizedInt32();
                point.V = new float[2];
                point.V[0] = ix;
                point.V[1] = iy;
                ix += 1;
                if (ix == XCount)
                {
                    ix = 0;
                    iy += 1;
                }
            }
        }
    }

    public class Point
    {
        /// <summary>
        ///     Index in Words
        /// </summary>
        public int WordIndex;

        public int GroupId_ToDisplay = (int)PointGroupId_ToDisplay.None;

        /// <summary>
        ///     |iX, iY| vector
        /// </summary>
        /// <remarks>Otimized for calculation</remarks>
        public float[] V = null!;

        /// <summary>
        ///     Top N point refs (ordered by proximity)
        ///     (Proximity, Point)
        /// </summary>
        public (float, Point)[]? Temp_TopProxPoints;

        /// <summary>
        ///     Top N primary point refs (ordered by proximity)
        ///     (Proximity, Point)
        ///     Not null only for primary words.
        /// </summary>
        public (float, Point)[]? Temp_TopProxPrimaryPoints;

        public void CopyData(Point that)
        {
            WordIndex = that.WordIndex;
            GroupId_ToDisplay = that.GroupId_ToDisplay;
            V[0] = that.V[0];
            V[1] = that.V[1];
            Temp_TopProxPoints = that.Temp_TopProxPoints;
            Temp_TopProxPrimaryPoints = that.Temp_TopProxPrimaryPoints;
        }
    }
}

public enum CortexDisplayType
{
    GroupId_ToDisplay = 0,
    Spot,
}

public enum DotProductVariant
{
    All = 0,
    PrimaryOnly,
    SecondaryOnly,
}

public enum PointGroupId_ToDisplay
{
    None = 0,
    // 1-9 reserved for different colored groups
    PrimaryPoint = 10,
    MainPoint1 = 12,
    PrimaryPoint_Selected1 = 13,
    SecondaryPoint_Selected1 = 14,
    PrimaryAndSecondaryPoint_Selected1 = 15,
    //MainPoint2 = 15,
    //PrimaryPoint_Selected2 = 16,
    //SecondaryPoint_Selected2 = 17,
}

public class WordsNewEmbeddings
{
    /// <summary>
    ///     [Словоформа, DiscreteVector Index]
    /// </summary>
    public CaseInsensitiveDictionary<int> Words = new();
}



//Parallel.ForEach<Point?, double>(
//    Cortex.Array, // source collection
//    () => 0.0, // method to initialize the local variable
//    (point2, loopState, localPointEnergy) => // method invoked by the loop on each iteration
//    {
//        if (point2 is null || point2.Id == point.Id)
//            return localPointEnergy; // value to be passed to next iteration

//        var proxWord = ProxWords[point2.Id + idBias];
//        if (proxWord > 0)
//        {
//            _v2[0] = point2.iX;
//            _v2[1] = point2.iY;                        

//            localPointEnergy += proxWord * TensorPrimitives.Distance(_v1, _v2);
//        }
//        return localPointEnergy; // value to be passed to next iteration
//    }, 
//    localPointEnergy => // Method to be executed when each partition has completed.
//    { 
//        lock (pointEnergySyncRoot)
//        {
//            pointEnergy += localPointEnergy;
//        }                    
//    });
//    