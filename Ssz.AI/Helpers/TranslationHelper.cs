using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Ssz.Utils.Logging;
using System;
using System.Net.Http;
using System.Threading.Tasks;

namespace Ssz.AI.Helpers;

public class TranslationHelper
{
    private static readonly HttpClient HttpClient = new HttpClient();

    public static async Task<string> TranslateWordAsync(string word, ILoggersSet loggersSet)
    {
        try
        {
            // Неофициальный endpoint Google Translate
            string url = $"https://translate.googleapis.com/translate_a/single?client=gtx&sl=ru&tl=en&dt=t&q={Uri.EscapeDataString(word)}";
            var response = await HttpClient.GetStringAsync(url);

            // Парсинг JSON-ответа (первый перевод)
            dynamic json = JsonConvert.DeserializeObject(response)!;
            return json[0][0][0].ToString();
        }
        catch (Exception ex)
        {
            loggersSet.UserFriendlyLogger.LogWarning($"Ошибка перевода для слова '{word}': {ex.Message}");
            return "{EN_translation}"; // Плейсхолдер на случай ошибки
        }
    }
}