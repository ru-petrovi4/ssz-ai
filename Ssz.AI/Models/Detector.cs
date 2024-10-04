using OpenCvSharp;
using Ssz.AI.Helpers;
using System;

namespace Ssz.AI.Models
{
    public class Detector
    {
        public double CenterX { get; init; }
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

        public bool IsActivated(GradientInPoint[,] gradientMatrix)
        {
            (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(CenterX, CenterY, gradientMatrix);            

            bool activated = (magnitude >= GradientMagnitudeLowLimit) && (magnitude < GradientMagnitudeHighLimit);
            if (!activated)
                return false;    

            if (GradientAngleHighLimit > GradientAngleLowLimit)
                activated = (angle >= GradientAngleLowLimit) && (angle < GradientAngleHighLimit);
            else
                activated = (angle >= GradientAngleLowLimit) || (angle < GradientAngleHighLimit);
            return activated;
        }
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