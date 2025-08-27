using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Logging;
using OfficeOpenXml;
using Ssz.Utils;
using Ssz.Utils.Logging;
using System;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    public partial class Model01
    {
        private void CompareOldAndNewPhraseEmbeddings(ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            FileInfo resultTemplateFileInfo = new(Path.Combine(programDataDirectoryFullName, @"df.xlsx"));
            FileInfo resultFileInfo = new(Path.Combine(programDataDirectoryFullName, @"df_out.xlsx"));
            if (resultFileInfo.Exists)
                resultFileInfo.Delete();

            using (var package = new ExcelPackage(resultTemplateFileInfo, true))
            {
                var sentence1Range = package.Workbook.Names["sentence1"];                
                var resultRange = package.Workbook.Names["result"];

                var sentence1Column = sentence1Range.Start.Column;                
                var resultColumn = resultRange.Start.Column;
                
                for (int row = sentence1Range.Start.Row + 1; ; row += 1)
                {
                    string sentence1 = new Any(sentence1Range.Worksheet.Cells[row, sentence1Column].Value).ValueAsString(false);
                    string sentence2 = new Any(sentence1Range.Worksheet.Cells[row, sentence1Column + 1].Value).ValueAsString(false);
                    if (string.IsNullOrEmpty(sentence1))
                        break;
                    var sentence1Embeding = GetEmbeddingForPhrase(sentence1);
                    var sentence2Embeding = GetEmbeddingForPhrase(sentence2);
                    NormalizePhraseEmbeding(sentence1Embeding);
                    NormalizePhraseEmbeding(sentence2Embeding);
                    var result = TensorPrimitives.Dot(sentence1Embeding, sentence2Embeding);
                    resultRange.Worksheet.Cells[row, resultColumn].Value = $"{result:F02}";
                }

                package.SaveAs(resultFileInfo);
            }


            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation($"{nameof(CompareOldAndNewPhraseEmbeddings)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");
        }

        private void NormalizePhraseEmbeding(float[] phraseEmbeding)
        {
            //for (int i = 0; i < phraseEmbeding.Length; i += 1)
            //{
            //    phraseEmbeding[i] = phraseEmbeding[i] > 0.1f ? 1.0f : 0.0f;
            //}
            float norm = TensorPrimitives.Norm(phraseEmbeding);
            TensorPrimitives.Divide(phraseEmbeding, norm, phraseEmbeding);
        }
    }
}
