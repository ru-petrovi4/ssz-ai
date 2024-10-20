using OpenCvSharp;
using Ssz.AI.Helpers;
using System;

namespace Ssz.AI.Models
{
    public class Detector
    {
        /// <summary>
        ///     Минимальная чувствительность к модулю градиента
        /// </summary>
        public const double GradientMagnitudeMinimum = 5.0;

        /// <summary>
        ///     [0..MNISTImageWidth]
        /// </summary>
        public double CenterX { get; init; }

        /// <summary>
        ///     [0..MNISTImageHeight]
        /// </summary>
        public double CenterY { get; init; }
        //public double Width { get; init; }
        public double GradientMagnitudeLowLimit { get; init; }
        public double GradientMagnitudeHighLimit { get; init; }
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleLowLimit { get; init; }
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleHighLimit { get; init; }

        public int BitIndexInHash;

        public bool GetIsActivated(GradientInPoint[,] gradientMatrix)
        {
            (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(CenterX, CenterY, gradientMatrix);            

            if (magnitude < GradientMagnitudeMinimum) 
                return false;

            bool activated = (magnitude >= GradientMagnitudeLowLimit) && (magnitude < GradientMagnitudeHighLimit);
            if (!activated)
                return false;    

            if (GradientAngleHighLimit > GradientAngleLowLimit)
                activated = (angle >= GradientAngleLowLimit) && (angle < GradientAngleHighLimit);
            else
                activated = (angle >= GradientAngleLowLimit) || (angle < GradientAngleHighLimit);
            return activated;
        }

        public bool Temp_IsActivated;
    }
}


//public Detector(double centerX, double centerY, double width, double angleLoLimit, double angleRange, double gradientMagnitudeLoLimit, double gradientMagnitudeRange)
//{
//    CenterX = centerX;
//    CenterY = centerY;
//    Width = width;

//    AngleLoLimit = angleLoLimit;
//    AngleRange = angleRange;
//    GradientMagnitudeLoLimit = gradientMagnitudeLoLimit;
//    GradientMagnitudeRange = gradientMagnitudeRange;
//}