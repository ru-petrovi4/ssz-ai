using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    public static partial class ModelHelper
    {
        #region public functions

        /// <summary>
        ///     
        /// </summary>
        /// <param name="i">Количество совпадений</param>
        /// <param name="n">Количество единиц</param>
        /// <param name="m">Длина вектора</param>
        /// <returns>Вероятность <paramref name="i"/> совпадений в случайных векторах</returns>
        public static double GetProbability(int i, int n, int m)
        {
            // Если невозможно расположить оставшиеся элементы без пересечений то, верятность 0
            if ((n - i) * 2 > m - i) 
                return 0.0;
            var общее_количество_комбинаций = Factorial(m, n) / Factorial(n);
            var общее_количество_комбинаций_двух_векторов = общее_количество_комбинаций * общее_количество_комбинаций;

            // Количество комбинаций из i совпадений
            var количество_комбинаций1 = Factorial(m, i) / Factorial(i);

            // Количество комбинаций оставшихся элементов 1-го вектора
            var количество_комбинаций2 = Factorial(m - i, n - i) / Factorial(n - i);

            // Количество комбинаций оставшихся элементов 2-го вектора, чтобы не было совпадений
            var количество_комбинаций3 = Factorial(m - n, n - i) / Factorial(n - i);

            return (double)(количество_комбинаций1 * количество_комбинаций2 * количество_комбинаций3) / (double)общее_количество_комбинаций_двух_векторов;
        }

        #endregion

        #region private functions

        private static BigInteger Factorial(int value)
        {
            BigInteger result = 1;
            for (int i = value; i > 1; i -= 1)
                result *= i;
            return result;
        }

        private static BigInteger Factorial(int value, int count)
        {
            BigInteger result = 1;
            for (int i = 0; i < count; i += 1)
            {
                result *= value;
                value -= 1;
            }
            return result;
        }
        
        #endregion        
    }
}
