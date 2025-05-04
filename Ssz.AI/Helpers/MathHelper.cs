using Ssz.AI.Models;
using System;

namespace Ssz.AI.Helpers
{
    public static class MathHelper
    {
        /// <summary>
        ///     Radians [-pi, pi], Degrees [0..360)
        /// </summary>
        /// <param name="radians"></param>
        /// <returns></returns>
        public static double RadiansToDegrees(double radians)
        {
            double degrees = 180 * radians / Math.PI;
            if (degrees < 0)
            {
                if (degrees < -0.000001)
                    degrees += 360;
                else
                    degrees = 0.0;
            }                
            return degrees;
        }

        /// <summary>
        ///     Degrees [0..360], Radians [-pi, pi)
        /// </summary>
        /// <param name="degrees"></param>
        /// <returns></returns>
        public static double DegreesToRadians(double degrees)
        {
            double radians = Math.PI * degrees / 180;
            if (radians > Math.PI - 0.000001)
            {
                if (radians > Math.PI)
                    radians -= 2 * Math.PI;
                else
                    radians = -Math.PI;
            }                
            return radians;
        }

        public static GradientInPoint GetInterpolatedGradient(double centerX, double centerY, DenseMatrix<GradientInPoint> gradientMatrix)
        {   
            int x = (int)centerX;
            int y = (int)centerY;
            if (x < 0 ||
                    y < 0 ||
                    x + 1 >= gradientMatrix.Dimensions[0] ||
                    y + 1 >= gradientMatrix.Dimensions[1])
                return new();

            double tx = centerX - x;
            double ty = centerY - y;
            double gradX = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradX +
                tx * (1 - ty) * gradientMatrix[x + 1, y].GradX +
                (1 - tx) * ty * gradientMatrix[x, y + 1].GradX +
                tx * ty * gradientMatrix[x + 1, y + 1].GradX;
            double gradY = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradY +
                tx * (1 - ty) * gradientMatrix[x + 1, y].GradY +
                (1 - tx) * ty * gradientMatrix[x, y + 1].GradY +
                tx * ty * gradientMatrix[x + 1, y + 1].GradY;

            double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);

            double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

            if (angle > Math.PI - 0.000001)
                angle = -Math.PI;

            return new GradientInPoint
            {
                GradX = gradX,
                GradY = gradY,
                Angle = angle,
                Magnitude = magnitude
            };
        }

        /// <summary>
        ///     Returns [-pi, pi)
        /// </summary>
        /// <param name="centerX"></param>
        /// <param name="centerY"></param>
        /// <param name="gradientMatrix"></param>
        /// <returns></returns>
        public static (double magnitude, double angle) GetInterpolatedGradient_Obsolete(double centerX, double centerY, GradientInPoint[,] gradientMatrix)
        {
            int x = (int)centerX;
            int y = (int)centerY;
            if (x < 0 ||
                    y < 0 ||
                    x + 1 >= gradientMatrix.GetLength(0) ||
                    y + 1 >= gradientMatrix.GetLength(1))
                return (0.0, 0.0);

            double tx = centerX - x;
            double ty = centerY - y;
            double gradX = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradX +
                tx * (1 - ty) * gradientMatrix[x + 1, y].GradX +
                (1 - tx) * ty * gradientMatrix[x, y + 1].GradX +
                tx * ty * gradientMatrix[x + 1, y + 1].GradX;
            double gradY = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradY +
                tx * (1 - ty) * gradientMatrix[x + 1, y].GradY +
                (1 - tx) * ty * gradientMatrix[x, y + 1].GradY +
                tx * ty * gradientMatrix[x + 1, y + 1].GradY;

            double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);

            double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

            if (angle > Math.PI - 0.000001)
                angle = -Math.PI;

            return (magnitude, angle);
        }
    }
}
