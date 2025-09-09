using System;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public static class Hungarian
{
    // Малый допуск для надёжной проверки «нуля» после численных преобразований.
    private const float Eps = 1e-6f;

    /// <summary>
    /// Оптимальное назначение (минимизация суммы стоимости) для квадратной матрицы cost размера N×N.
    /// Возвращает массив size N: assignRowToCol[i] = j (строка i сопоставлена столбцу j).
    /// </summary>
    public static int[] HungarianAlgorithm(MatrixFloat cost)
    {
        int n = cost.Dimensions[0]; // Предполагаем квадратную матрицу N x N

        // 1) Копируем в быстрый строко-ориентированный буфер (row-major) для векторных операций по строкам.
        //    MatrixFloat хранится в column-major (i + j * rows), поэтому спроецируем в row-major: rm[i * n + j].
        var rm = new float[n * n];
        for (int i = 0; i < n; ++i)
        {
            int baseRow = i * n;
            for (int j = 0; j < n; ++j)
            {
                rm[baseRow + j] = cost[i, j];
            }
        }

        // Вспомогательные массивы для алгоритма:
        // stars: звездный ноль (назначение): по строкам и по столбцам
        var starColOfRow = new int[n];   // -1 если в строке нет звезды
        var starRowOfCol = new int[n];   // -1 если в столбце нет звезды
        Array.Fill(starColOfRow, -1);
        Array.Fill(starRowOfCol, -1);

        // primes: зачёркнутые нули (prime): хранить по строкам столбец прайма
        var primeColOfRow = new int[n];  // -1 если прайма нет
        Array.Fill(primeColOfRow, -1);

        // Покрытия строк и столбцов
        var coverRow = new bool[n];
        var coverCol = new bool[n];

        // Локальные функции-доступы для rm
        //static ReadOnlySpan<float> RowRO(Span<float> buf, int n, int i) => new ReadOnlySpan<float>(buf.ToArray(), i * n, n);
        // Но выше копирует, что неэффективно; сделаем нормальный Span:
        static Span<float> Row(Span<float> buf, int n, int i) => buf.Slice(i * n, n);
        static ReadOnlySpan<float> RowRO(ReadOnlySpan<float> buf, int n, int i) => buf.Slice(i * n, n);

        var rmSpan = rm.AsSpan();

        // 2) Строковая редукция: вычитаем минимумы по строкам, чтобы создать нули.
        for (int i = 0; i < n; ++i)
        {
            var row = RowRO(rm, n, i);
            float minVal = row[0];
            for (int j = 1; j < n; ++j)
                if (row[j] < minVal) 
                    minVal = row[j];

            if (minVal > 0)
            {
                // Вычитаем minVal из всей строки используя векторную примитивную операцию Add(Span, scalar, dst)
                var rowW = Row(rmSpan, n, i);
                // TensorPrimitives.Add(dst, -minVal, dst) эквивалент dst[k] += -minVal
                TensorPrimitives.Add(rowW, -minVal, rowW);
            }
        }

        // 3) Столбцовая редукция: вычитаем минимумы по столбцам.
        // Для столбцов нет непрерывности в row-major, применим обычный цикл.
        for (int j = 0; j < n; ++j)
        {
            float minVal = rm[j];
            for (int i = 1; i < n; ++i)
            {
                float v = rm[i * n + j];
                if (v < minVal) minVal = v;
            }
            if (minVal > 0)
            {
                for (int i = 0; i < n; ++i)
                {
                    rm[i * n + j] -= minVal;
                }
            }
        }

        // 4) Начальное «звездение» нулей: для каждого нулевого элемента, где строка и столбец ещё не заняты, ставим звезду.
        var rowHasStar = new bool[n];
        var colHasStar = new bool[n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (!rowHasStar[i] && !colHasStar[j] && MathF.Abs(rm[i * n + j]) <= Eps)
                {
                    starColOfRow[i] = j;
                    starRowOfCol[j] = i;
                    rowHasStar[i] = true;
                    colHasStar[j] = true;
                }
            }
        }

        // 5) Покрыть все столбцы, содержащие звёзды. Если покрытых столбцов n — готово.
        Array.Clear(coverRow, 0, n);
        Array.Clear(coverCol, 0, n);
        int coveredCols = 0;
        for (int j = 0; j < n; ++j)
        {
            if (starRowOfCol[j] != -1)
            {
                coverCol[j] = true;
                coveredCols++;
            }
        }
        if (coveredCols == n)
            return starColOfRow; // Оптимальное решение найдено

        // 6) Главный цикл: ищем непокрытые нули, строим чередующиеся пути (prime-star), корректируем покрытие и матрицу.
        while (true)
        {
            // Ищем непокрытый ноль; если не нашли — модифицируем матрицу и повторяем поиск.
            int zRow = -1, zCol = -1;
            
            while (true)
            {
                bool foundUncoveredZero = FindUncoveredZero(rm, n, coverRow, coverCol, out zRow, out zCol);

                if (!foundUncoveredZero)
                {
                    // 6.a) Нет непокрытых нулей — модифицируем матрицу:
                    // Находим минимальный «непокрытый» элемент h; вычитаем h из всех непокрытых,
                    // прибавляем h ко всем дважды покрытым (пересечение покрытых строк и покрытых столбцов).
                    float h = FindMinUncovered(rm, n, coverRow, coverCol);
                    // Вычитаем h из всех элементов непокрытых строк И непокрытых столбцов
                    for (int i = 0; i < n; ++i)
                    {
                        bool rowCovered = coverRow[i];
                        for (int j = 0; j < n; ++j)
                        {
                            bool colCovered = coverCol[j];
                            if (!rowCovered && !colCovered)
                            {
                                rm[i * n + j] -= h;
                            }
                            else if (rowCovered && colCovered)
                            {
                                rm[i * n + j] += h;
                            }
                        }
                    }
                    // После модификации снова пытаемся найти непокрытый ноль
                    continue;
                }

                // 6.b) Нашли непокрытый ноль (zRow, zCol); помечаем его праймом.
                primeColOfRow[zRow] = zCol;

                // Если в этой строке нет звезды — строим чередующийся путь и «перекладываем» звезды.
                if (starColOfRow[zRow] == -1)
                {
                    // Строим чередующийся путь (row0=zRow,col0=zCol), чередуя prime и star до упора.
                    AugmentPath(rm, n, zRow, zCol, starColOfRow, starRowOfCol, primeColOfRow);

                    // Снимаем все покрытия и удаляем все праймы.
                    Array.Clear(coverRow, 0, n);
                    Array.Clear(coverCol, 0, n);
                    Array.Fill(primeColOfRow, -1);

                    // Покрыть все столбцы, содержащие звёзды; если все n покрыты — конец.
                    coveredCols = 0;
                    for (int j = 0; j < n; ++j)
                    {
                        if (starRowOfCol[j] != -1)
                        {
                            coverCol[j] = true;
                            coveredCols++;
                        }
                    }
                    if (coveredCols == n)
                        return starColOfRow;

                    // Иначе продолжаем внешний цикл поиска непокрытого нуля
                    break;
                }
                else
                {
                    // Иначе — в строке есть звезда: покрываем строку, снимаем покрытие со столбца со звездой и продолжаем поиск.
                    int starCol = starColOfRow[zRow];
                    coverRow[zRow] = true;
                    coverCol[starCol] = false;
                    // Продолжаем while(true) поиск следующего непокрытого нуля
                }
            }
        }

        // Локальные функции:

        bool FindUncoveredZero(float[] a, int n, bool[] coverRow, bool[] coverCol, out int r, out int c)
        {
            for (int i = 0; i < n; ++i)
            {
                if (coverRow[i]) continue;
                int baseRow = i * n;
                for (int j = 0; j < n; ++j)
                {
                    if (coverCol[j]) continue;
                    if (MathF.Abs(a[baseRow + j]) <= Eps)
                    {
                        r = i; c = j; return true;
                    }
                }
            }
            r = -1; c = -1; return false;
        }

        float FindMinUncovered(float[] a, int n, bool[] coverRow, bool[] coverCol)
        {
            float minVal = float.PositiveInfinity;
            for (int i = 0; i < n; ++i)
            {
                if (coverRow[i]) continue;
                int baseRow = i * n;
                for (int j = 0; j < n; ++j)
                {
                    if (coverCol[j]) continue;
                    float v = a[baseRow + j];
                    if (v < minVal) minVal = v;
                }
            }
            return minVal;
        }

        void AugmentPath(
            float[] a, int n,
            int row, int col,
            int[] starColOfRow, int[] starRowOfCol,
            int[] primeColOfRow)
        {
            // Чередующийся путь: начинаем с (row,col) — это прайм; далее ищем в этом столбце звезду и продолжаем.
            // Будем «перекладывать»: звезду в позициях праймов и удалять звезды в предыдущих позициях.
            // Для эффективности используем временный список пар (r,c), по которым нужно «переключить» звезду.
            Span<(int r, int c)> path = stackalloc (int r, int c)[2 * n];
            int pathLen = 0;
            path[pathLen++] = (row, col);

            while (true)
            {
                int rStar = starRowOfCol[col];
                if (rStar == -1)
                    break; // нет звезды в столбце — путь закончен

                // добавляем звезду (rStar, col) и найдём прайм в строке rStar
                path[pathLen++] = (rStar, col);
                int cPrime = primeColOfRow[rStar];
                path[pathLen++] = (rStar, cPrime);
                row = rStar;
                col = cPrime;
            }

            // Переключаем: клетки с праймами становятся звёздами; клетки со звёздами — снимаются.
            for (int k = 0; k < pathLen; ++k)
            {
                var (r, c) = path[k];
                if (starColOfRow[r] == c)
                {
                    // была звезда — снимаем
                    starColOfRow[r] = -1;
                    starRowOfCol[c] = -1;
                }
                else
                {
                    // был прайм — ставим звезду
                    // Если в строке r уже была звезда, она должна быть снята ранее по пути
                    starColOfRow[r] = c;
                    starRowOfCol[c] = r;
                }
            }
        }
    }
}
