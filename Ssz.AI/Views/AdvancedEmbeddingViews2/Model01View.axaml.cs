using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Logging;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel2;
using Avalonia.Threading;
using System.Threading;

namespace Ssz.AI.Views.AdvancedEmbeddingViews2;

public partial class Model01View : UserControl
{
    public Model01View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        var constants = Model01.Constants;
        SetDataToControls(constants);

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(constants);        
        LevelScrollBar3.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(constants);

        Reset();
        Refresh_ImagesSet();

        //Model = new Model01();
        //Task.Run(() =>
        //{
        //    Model.StemInputText();
        //});
    }

    public Model01 Model = null!;

    private void SetDataToControls(Model01.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;        
        LevelScrollBar3.Value = constants.K3;
        LevelScrollBar4.Value = constants.K4;        

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = constants.PositiveK[0];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = constants.PositiveK[1];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = constants.PositiveK[2];
        
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = constants.NegativeK[0];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = constants.NegativeK[1];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = constants.NegativeK[2];        
    }

    private void GetDataFromControls(Model01.ModelConstants constants)
    {
        constants.K0 = (float)LevelScrollBar0.Value;
        constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;        
        constants.K3 = (float)LevelScrollBar3.Value;
        constants.K4 = (float)LevelScrollBar4.Value;        

        constants.PositiveK[0] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value;
        constants.PositiveK[1] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value;
        constants.PositiveK[2] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value;        

        constants.NegativeK[0] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value;
        constants.NegativeK[1] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value;
        constants.NegativeK[2] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value;        
    }

    #region Buttons Handlers

    private void Reset_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();

        Refresh_ImagesSet();
    }

    private void StopLongOperation_OnClick(object? sender, RoutedEventArgs args)
    {
        _cancellationTokenSource?.Cancel();
    }

    private void ResetPhrasesIterator_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.InputCorpusData.CurrentCortexMemoryIndex = -1;

        Refresh_ImagesSet();
    }

    private void ShowNextPhrase_OnClick(object? sender, RoutedEventArgs args)
    {
        if (Model.InputCorpusData.CurrentCortexMemoryIndex >= Model.InputCorpusData.CortexMemories.Count - 1)
            return;

        Model.InputCorpusData.CurrentCortexMemoryIndex += 1;

        Model.Cortex.Calculate_CurrentCortexMemory(Model.InputCorpusData, _random);

        Refresh_ImagesSet();
    }

    private void ResetWordsIterator_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.InputCorpusData.Current_OrderedWords_Index = -1;

        Refresh_ImagesSet();
    }

    private void ShowNextWord_OnClick(object? sender, RoutedEventArgs args)
    {
        if (Model.InputCorpusData.Current_OrderedWords_Index >= Model.Cortex.Words.Count - 1)
            return;

        Model.InputCorpusData.Current_OrderedWords_Index += 1;

        Model.Cortex.Calculate_CurrentWord(Model.InputCorpusData, _random);

        Refresh_ImagesSet();
    }

    private void GenerateCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Cortex.GenerateOwnedData(_random);
        Model.Cortex.Prepare();
    }

    private void LoadInputCorpusData_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.InputCorpusData = InputCorpusDataHelper.GetInputCorpusData(
                    Model.Cortex.Words,
                    _random,
                    Model01.Constants.DiscreteVectorLength,
                    Model01.Constants.DiscreteRandomVector_PrimaryBitsCount,
                    Model.LoggersSet.LoggerAndUserFriendlyLogger);        
    }

    private void SaveCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Helpers.SerializationHelper.SaveToFile(Model01.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);
    }

    private void LoadCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);
        Model.Cortex.Prepare();
    }

    private void PutPhrase_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Calculate_PutPhrases_BasedOnSuperActivity(1, _random);

        Refresh_ImagesSet();
    }

    private void PutPhrases2000_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Calculate_PutPhrases_BasedOnSuperActivity(2000, _random);

        Refresh_ImagesSet();
    }

    private void PutPhrases5K_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Calculate_PutPhrases_BasedOnSuperActivity(5000, _random);

        Refresh_ImagesSet();
    }

    private void PutPhrases10K_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Calculate_PutPhrases_BasedOnSuperActivity(10000, _random);

        Refresh_ImagesSet();
    }

    private async void ReorderPhrases1Epoch_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        await Model.ReorderPhrases1Epoch_BasedOnSuperActivityAsync(1, _random, CancellationToken.None, async () =>
        {
            Refresh_ImagesSet();
            await Task.Delay(50);
        });

        Refresh_ImagesSet();
    }

    private async void StartReorderPhrasesAll_BasedOnSuperActivity_OnClick(object? sender, RoutedEventArgs args)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        await Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderPhrases1Epoch_BasedOnSuperActivityAsync Started.");

                await Model.ReorderPhrases1Epoch_BasedOnSuperActivityAsync(100, _random, cancellationToken, () =>
                {
                    Dispatcher.UIThread.Invoke(() =>
                    {
                        Refresh_ImagesSet();
                    });
                    return Task.CompletedTask;
                });
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderPhrases1Epoch_BasedOnSuperActivityAsync Cancelled.");
            }

            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderPhrases1Epoch_BasedOnSuperActivityAsync Finished.");
        });

        Refresh_ImagesSet();
    }

    private async void DoScript0_OnClick(object? sender, RoutedEventArgs args)
    {        
        await Task.Run(async () =>
        {
            for (; ; )
            {
                bool finished = Model.Calculate_PutPhrases_BasedOnSuperActivity(5000, _random);

                await Model.ReorderPhrases1Epoch_BasedOnSuperActivityAsync(7, _random, CancellationToken.None, () =>
                {
                    Dispatcher.UIThread.Invoke(() =>
                    {
                        Refresh_ImagesSet();
                    });
                    return Task.CompletedTask;
                });

                if (finished)
                    break;
            }
        });

        Refresh_ImagesSet();
    }

    private async void DoScript1_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);
        
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);
        Model.Cortex.Prepare();

        Model.Cortex.Calculate_Words_DiscreteOptimizedVectors(_random);
        Helpers.SerializationHelper.SaveToFile(Model01.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);

        Refresh_ImagesSet();
    }

    private async void DoScript2_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);

        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);

        Model.Cortex.Calculate_Words_DiscreteOptimizedVectors_Metrics(_random, Model.LoggersSet.LoggerAndUserFriendlyLogger);
    }

    private async void PutPhrasesAll_Randomly_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);        

        Model.Calculate_PutPhrases_Randomly(Int32.MaxValue, _random);
    }

    private async void StartReoderPhrases_BasedOnCodingDecoding_OnClick(object? sender, RoutedEventArgs args)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;
        await Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Calculate_ReorderPhrases_BasedOnCodingDecoding Started.");

                Model.InputCorpusData = InputCorpusDataHelper.GetInputCorpusData(
                    Model.Cortex.Words,
                    _random, 
                    Model01.Constants.DiscreteVectorLength, 
                    Model01.Constants.DiscreteRandomVector_PrimaryBitsCount,
                    Model.LoggersSet.LoggerAndUserFriendlyLogger);

                await Model.Cortex.Prepare_Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
                    Model.InputCorpusData.CortexMemories,
                    _random);

                await Model.Cortex.Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
                    cortexMemories_EpochCount: 1000000,
                    miniColumns_EpochCount: 1000000,
                    _random,
                    cancellationToken,
                    () =>
                    {
                        Dispatcher.UIThread.Invoke(() =>
                        {
                            Refresh_ImagesSet();
                        });
                        return Task.CompletedTask;
                    });
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Calculate_ReorderPhrases_BasedOnCodingDecoding Cancelled.");
            }
            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Calculate_ReorderPhrases_BasedOnCodingDecoding Finished.");
        });
    }

    private async void DoScript3_OnClick(object? sender, RoutedEventArgs args)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        await Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Script3 Started.");

                Model.InputCorpusData = InputCorpusDataHelper.GetInputCorpusData(
                    Model.Cortex.Words,
                    _random,
                    Model01.Constants.DiscreteVectorLength,
                    Model01.Constants.DiscreteRandomVector_PrimaryBitsCount,
                    Model.LoggersSet.LoggerAndUserFriendlyLogger);

                for (int onesCount = 1; onesCount < 30; onesCount += 1)
                {
                    Model01.Constants.DiscreteOptimizedVector_PrimaryBitsCount = onesCount;

                    await Model.Cortex.Prepare_Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
                        Model.InputCorpusData.CortexMemories,
                        _random);

                    await Model.Cortex.Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
                        cortexMemories_EpochCount: 1,
                        miniColumns_EpochCount: 0,
                        _random,
                        cancellationToken,
                        () =>
                        {
                            Dispatcher.UIThread.Invoke(() =>
                            {
                                Refresh_ImagesSet();
                            });
                            return Task.CompletedTask;
                        });
                }
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Script3 Cancelled.");
            }

            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("Script3 Finished.");
        });

        Refresh_ImagesSet();
    }

    #endregion

    private void Reset()
    {
        var constants = Model01.Constants;
        GetDataFromControls(constants);

        _random = new Random(41);

        Model = new Model01();

        Model.Cortex = new Models.AdvancedEmbeddingModel2.Cortex(Model01.Constants, Model.LoggersSet.LoggerAndUserFriendlyLogger);        
    }

    private void Refresh_ImagesSet()
    {
        if (Model.Cortex.MiniColumns is null)
            return;

        ImagesSet1_TextBlock.Text = Model.Cortex.Temp_InputCurrentDesc;
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs();

        if (Model.InputCorpusData is null)
            return;

        Model.Cortex.Calculate_CurrentWord(Model.InputCorpusData, _random);

        ImagesSet2_TextBlock.Text = Model.Cortex.Temp_InputCurrentDesc;
        ImagesSet2.MainItemsControl.ItemsSource = Model.GetImageWithDescs();
    }

    private Random _random = null!;

    private CancellationTokenSource? _cancellationTokenSource;
}