using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using System;
using System.IO;
using System.Text;

namespace Ssz.AI;

public partial class DebugWindow : Window
{
    #region construction and destruction

    public DebugWindow()
    {
        InitializeComponent();
    }

    #endregion

    #region public functions

    /// <summary>
    ///     Created if needed
    /// </summary>
    public static DebugWindow Instance
    {
        get
        {
            if (_instance is null)
            {
                Dispatcher.UIThread.Invoke(() =>
                {
                    _instance = new DebugWindow();
                    _instance.Closed += (sender, args) => { _instance = null; };
                    _instance.Show();
                });                
            }

            return _instance!;
        }
    }

    public static bool IsWindowExists => _instance is not null;

    public void Clear()
    {
        Dispatcher.UIThread.Invoke(() =>
        {
            MainTextEditor.Clear();
        });
    }

    public void AddLine(string line)
    {
        Dispatcher.UIThread.Invoke(() =>
        {
            var scrollViewer = MainTextEditor.FindDescendantOfType<ScrollViewer>()!;
            bool atBottom = false;
            if (scrollViewer.Offset.Y > scrollViewer.ScrollBarMaximum.Y * 0.95)
                atBottom = true;

            MainTextEditor.Text += line + "\n";  

            if (atBottom)
                MainTextEditor.ScrollToEnd();
        });        
    }

    public static void AddLine(LogLevel logLevel, EventId eventId, string message)
    {
        Instance.AddLine(message);
    }

    public void SaveToFile()
    {
        File.WriteAllText($"Data\\Script\\Log_{new Any(DateTime.Now).ValueAsString(false)}.csv", MainTextEditor.Text, new UTF8Encoding(true));        
    }

    #endregion

    #region private fields

    private static DebugWindow? _instance;

    #endregion
}