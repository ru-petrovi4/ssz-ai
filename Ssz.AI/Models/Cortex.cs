using System;
using System.Collections.Generic;
using System.IO;

namespace Ssz.AI.Models
{
    public class Cortex
    {
        public Cortex(
            int cortexWidth, 
            int cortexHeight, 
            int imageWidth, 
            int imageHeight,
            int miniColumnVisibleDetectorsCount,
            double detectorArea,
            int hashLength,
            List<Detector> detectors)
        {
            Random random = new();

            double visibleRadius = Math.Sqrt(miniColumnVisibleDetectorsCount * detectorArea / Math.PI);

            MiniColumns = new MiniColumn[cortexWidth, cortexHeight];
            double minCenterX = visibleRadius;
            double maxCenterX = imageWidth - visibleRadius;
            double deltaCenterX = (maxCenterX - minCenterX) / (cortexWidth - 1);
            double minCenterY = visibleRadius;
            double maxCenterY = imageHeight - visibleRadius;
            double deltaCenterY = (maxCenterY - minCenterY) / (cortexHeight - 1);
            for (int cy = 0; cy < cortexHeight; cy += 1)
            {
                for (int cx = 0; cx < cortexWidth; cx += 1)
                {
                    MiniColumn miniColumn = new MiniColumn(
                        miniColumnVisibleDetectorsCount,
                        minCenterX + cx * deltaCenterX,
                        minCenterY + cy * deltaCenterY);

                    foreach (var detector in detectors)
                    {
                        double r = Math.Sqrt(Math.Pow(detector.CenterX - miniColumn.CenterX, 2) + Math.Pow(detector.CenterY - miniColumn.CenterY, 2));
                        if (r < visibleRadius)
                            miniColumn.Detectors.Add(detector);
                    }

                    MiniColumns[cx, cy] = miniColumn;
                }
            }

            HashFunction = new int[imageWidth, imageHeight];
            for (int y = 0; y < imageHeight; y += 1)
            {
                for (int x = 0; x < imageWidth; x += 1)
                {
                    HashFunction[x, y] = random.Next(hashLength);
                }
            }
        }

        public MiniColumn[,] MiniColumns { get; }

        public int[,] HashFunction { get; }
    }

    public readonly struct MiniColumn
    {
        public MiniColumn(int miniColumnVisibleDetectorsCount, double centerX, double centerY)
        {
            Detectors = new(miniColumnVisibleDetectorsCount);
            CenterX = centerX;
            CenterY = centerY;
        }

        public readonly List<Detector> Detectors;

        public readonly double CenterX;

        public readonly double CenterY;
    }
}
