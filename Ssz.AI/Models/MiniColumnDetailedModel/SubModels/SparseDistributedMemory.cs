using System;
using System.Collections;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel.SubModels;

public class SparseDistributedMemory
{
    // --------------------------------------------------------------------
    // 1. СТРУКТУРА ДАННЫХ
    // --------------------------------------------------------------------

    /// <summary>
    /// Класс описывает одну физическую ячейку памяти (Hard Location).    
    /// </summary>
    private class HardLocation
    {        
        public bool[] Address { get; }

        // Счётчики для каждого бита данных.
        // Если мы записываем бит '1', счетчик растет. Если '0' - падает.
        public int[] DataBitsCounters { get; }

        public HardLocation(int addressBitsCount, int dataBitsCount, Random rng)
        {
            Address = new bool[addressBitsCount];
            // Инициализируем адрес случайными битами
            for (int i = 0; i < addressBitsCount; i++)
            {
                Address[i] = rng.Next(2) == 1; // 50% шанс получить true
            }

            DataBitsCounters = new int[dataBitsCount]; // По умолчанию заполнен нулями
        }
    }

    private readonly HardLocation[] _hardLocations;
    private readonly int _radius;

    public int DataBitsCount { get; }

    public SparseDistributedMemory(int addressBitsCount, int dataBitsCount, int hardLocationCount, int radius, int? seed = null)
    {
        DataBitsCount = dataBitsCount;
        _radius = radius;

        var random = seed.HasValue ? new Random(seed.Value) : new Random();

        _hardLocations = new HardLocation[hardLocationCount];
        for (int i = 0; i < hardLocationCount; i++)
        {
            _hardLocations[i] = new HardLocation(addressBitsCount, dataBitsCount, random);
        }
    }

    // --------------------------------------------------------------------
    // 2. АЛГОРИТМ ЗАПИСИ
    // --------------------------------------------------------------------

    /// <summary>
    /// Запись данных по адресу.    
    /// </summary>
    public void Write(bool[] address, bool[] data)
    {        
        // Шаг 1: Подготавливаем массив "сдвигов" (+1 или -1) для данных.
        // Это позволит нам прибавлять их разом с помощью векторных инструкций.
        int[] increments = new int[DataBitsCount];
        for (int i = 0; i < DataBitsCount; i++)
        {
            increments[i] = data[i] ? 1 : -1;
        }

        // Шаг 2: Ищем ячейки-соседи и обновляем их счётчики
        foreach (var hardLocation in _hardLocations)
        {
            // Если дистанция Хэмминга (число отличающихся бит) в пределах радиуса
            if (HammingDistance(address, hardLocation.Address) <= _radius)
            {
                // Ячейка активируется: прибавляем паттерн к её счётчикам
                AddCounters(hardLocation.DataBitsCounters, increments);
            }
        }
    }

    // --------------------------------------------------------------------
    // 3. АЛГОРИТМ ЧТЕНИЯ
    // --------------------------------------------------------------------

    /// <summary>
    /// Чтение данных по заданному (возможно, зашумленному) адресу.
    /// </summary>
    public bool[] Read(bool[] addressBytes)
    {
        // Здесь мы будем накапливать сумму счетчиков со всех сработавших ячеек
        int[] sumOfCounters = new int[DataBitsCount];

        // Шаг 1: Ищем активные ячейки-соседи (как при записи)
        foreach (var loc in _hardLocations)
        {
            if (HammingDistance(addressBytes, loc.Address) <= _radius)
            {
                // Шаг 2: Прибавляем счетчики активной ячейки к нашей общей сумме
                AddCounters(sumOfCounters, loc.DataBitsCounters);
            }
        }

        // Шаг 3: Пороговая функция (Thresholding)
        // Знак итоговой суммы для каждого бита говорит о том, какое значение "победило"
        bool[] result = new bool[DataBitsCount];
        for (int i = 0; i < DataBitsCount; i++)
        {
            // Если сумма > 0, восстанавливаем 1. Если <= 0, восстанавливаем 0.
            result[i] = sumOfCounters[i] > 0;
        }

        return result;
    }

    // --------------------------------------------------------------------
    // 4. ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    // --------------------------------------------------------------------

    /// <summary>
    /// Вычисляет расстояние Хэмминга (количество позиций, в которых биты различаются).    
    /// </summary>
    private int HammingDistance(bool[] a, bool[] b)
    {
        int distance = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) // Если биты не равны
            {
                distance++;
            }
        }
        return distance;
    }

    /// <summary>
    /// Безопасное векторное сложение двух массивов с использованием Vector<T> (SIMD).
    /// Позволяет складывать массивы "пачками", не прибегая к указателям (unsafe).
    /// </summary>
    private void AddCounters(int[] target, int[] source)
    {
        //int i = 0;

        //// Vector<int>.Count показывает, сколько чисел int влезает в процессорный регистр
        //// (обычно 4 для SSE, 8 для AVX2, 16 для AVX-512)
        //int step = Vector<int>.Count;

        //// Если процессор поддерживает SIMD
        //if (Vector.IsHardwareAccelerated)
        //{
        //    // Идем по массивам, пока хватает элементов для заполнения целого вектора
        //    while (i <= target.Length - step)
        //    {
        //        // Безопасно загружаем "пачку" чисел из массивов
        //        var vTarget = new Vector<int>(target, i);
        //        var vSource = new Vector<int>(source, i);

        //        // Векторное сложение: выполняется за 1 инструкцию процессора
        //        var vResult = Vector.Add(vTarget, vSource);

        //        // Копируем результат сложения обратно в целевой массив
        //        vResult.CopyTo(target, i);

        //        i += step;
        //    }
        //}

        // "Хвост" цикла: если длина массивов не кратна размеру вектора,
        // складываем оставшиеся элементы классическим способом
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += source[i];
        }
    }

    /// <summary>
    /// Утилита для конвертации Span<byte> в BitArray для удобства работы.
    /// </summary>
    private BitArray SpanToBitArray(ReadOnlySpan<byte> span)
    {
        var bitArray = new BitArray(span.Length);
        for (int i = 0; i < span.Length; i++)
        {
            bitArray[i] = span[i] != 0;
        }
        return bitArray;
    }
}

public static class TestProgram
{
    public static void TestMain()
    {
        int n = 256;   // AddressBits
        int m = 256;   // DataBits
        int hardLocations = 5000;
        int radius = 60;

        var sdm = new SparseDistributedMemory(n, m, hardLocations, radius);

        bool[] address = RandomBits(n);
        bool[] data = RandomBits(m);

        sdm.Write(address, data);

        // зашумим адрес ~5 бит
        bool[] noisy = (bool[])address.Clone();
        FlipRandomBits(noisy, 5);

        bool[] recalled = sdm.Read(noisy);

        int equalBits = 0;
        for (int i = 0; i < m; i++)
            if (recalled[i] == data[i]) equalBits++;

        Console.WriteLine($"Recovered {equalBits}/{m} bits correct");
    }

    private static readonly Random Rng = new Random();

    private static bool[] RandomBits(int len)
    {
        var v = new bool[len];
        for (int i = 0; i < len; i++) 
            v[i] = Rng.Next(2) == 1;
        return v;
    }

    private static void FlipRandomBits(bool[] v, int flips)
    {
        for (int k = 0; k < flips; k++)
        {
            int idx = Rng.Next(v.Length);
            v[idx] = !v[idx];
        }
    }
}