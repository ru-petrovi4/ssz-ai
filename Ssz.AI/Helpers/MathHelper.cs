using Ssz.AI.Models;
using System;

namespace Ssz.AI.Helpers
{
    public static class MathHelper
    {
        /// <summary>
        ///     Returns [-pi, pi)
        /// </summary>
        /// <param name="centerX"></param>
        /// <param name="centerY"></param>
        /// <param name="gradientMatrix"></param>
        /// <returns></returns>
        public static (double magnitude, double angle) GetInterpolatedGradient(double centerX, double centerY, GradientInPoint[,] gradientMatrix)
        {
            int x = (int)centerX;
            int y = (int)centerY;
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
