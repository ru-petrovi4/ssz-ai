using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Microsoft.Extensions.Logging;

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
                _instance = new DebugWindow();                
                _instance.Closed += (sender, args) => { _instance = null; };
                _instance.Show();
            }

            return _instance;
        }
    }

    public static bool IsWindowExists => _instance is not null;

    public void Clear()
    {
        MainTextEditor.Clear();
    }

    public void AddLine(string line)
    {
        MainTextEditor.Text += line + "\n";

        MainTextEditor.ScrollToEnd();
    }

    public static void AddLine(LogLevel logLevel, EventId eventId, string message)
    {
        Instance.AddLine(message);
    }

    #endregion

    #region private fields

    private static DebugWindow? _instance;

    #endregion
}