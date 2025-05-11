using Ssz.AI.Models;
using System;

namespace Ssz.AI.Helpers
{
    public static class MathHelper
    {
        /// <summary>
        ///     Radians to Degrees [0..360)
        /// </summary>
        /// <param name="radians"></param>
        /// <returns></returns>
        public static float RadiansToDegrees(float radians)
        {
            float degrees = 180 * radians / MathF.PI;                        
            return NormalizeAngleDegrees(degrees);
        }

        /// <summary>
        ///     Degrees [0..360)
        /// </summary>
        /// <param name="degrees"></param>
        /// <returns></returns>
        public static float NormalizeAngleDegrees(float degrees)
        {
            degrees = degrees % 360.0f;
            if (degrees < 0.000001f)
            {
                if (degrees < -0.000001f)
                    degrees += 360.0f;
                else
                    degrees = 0.0f;
            }
            return degrees;
        }

        /// <summary>
        ///     Degrees to Radians [-pi, pi)
        /// </summary>
        /// <param name="degrees"></param>
        /// <returns></returns>
        public static float DegreesToRadians(float degrees)
        {
            float radians = MathF.PI * degrees / 180.0f;            
            return NormalizeAngle(radians);
        }

        /// <summary>
        ///     Returns Radians [-pi, pi)
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static float NormalizeAngle(float radians)
        {
            radians = radians % (2.0f * MathF.PI);
            if (radians > MathF.PI - 0.00001f)
            {
                if (radians > MathF.PI + 0.00001f)
                    radians -= 2 * MathF.PI;
                else
                    radians = -MathF.PI;
            }
            else if (radians < -MathF.PI + 0.00001f)
            {
                if (radians < -MathF.PI - 0.00001f)
                    radians += 2 * MathF.PI;
                else
                    radians = -MathF.PI;
            }
            return radians;
        }

        public static float GetInterpolatedValue(float[] points, float x)
        {   
            if (x < 0.00001f)
                return points[0];
            int xi = (int)x;
            if (xi + 1 >= points.Length)
                return points[points.Length - 1];
            return points[xi] + (points[xi + 1] - points[xi]) * (x - (float)xi);
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
