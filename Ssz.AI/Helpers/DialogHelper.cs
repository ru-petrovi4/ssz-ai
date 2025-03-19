using Avalonia.Controls;
using Avalonia.Layout;
using DialogHostAvalonia;
using Ssz.Utils.Avalonia;
using System.Threading.Tasks;

namespace Ssz.AI.Helpers
{
    public static class DialogHelper
    {
        public static async Task<string?> GetValueFromUserAsync(string label)
        {
            return (await DialogHost.Show(new InputDialog(label))) as string;
            //Title = title;
            //var textBox = new TextBox();
            //var okButton = new Button { Content = "OK" };
            //var cancelButton = new Button { Content = "Cancel" };

            //okButton.Click += (_, _) => { Close(textBox.Text); };
            //cancelButton.Click += (_, _) => { Close(null); };

            //Content = new StackPanel
            //{
            //    Children =
            //{
            //    new TextBlock { Text = message },
            //    textBox,
            //    new StackPanel
            //    {
            //        Orientation = Orientation.Horizontal,
            //        Children = { okButton, cancelButton }
            //    }
            //}
            //};
        }

        //public class InputDialog : Dialog
        //{
        //    public InputDialog(string label)
        //    {                
        //        var textBox = new TextBox();
        //        var okButton = new Button { Content = "OK" };
        //        var cancelButton = new Button { Content = "Cancel" };

        //        okButton.Click += (_, _) => { Close(textBox.Text); };
        //        cancelButton.Click += (_, _) => { Close(null); };

        //        Content = new StackPanel
        //        {
        //            Children =
        //            {
        //                new TextBlock { Text = message },
        //                textBox,
        //                new StackPanel
        //                {
        //                    Orientation = Orientation.Horizontal,
        //                    Children = { okButton, cancelButton }
        //                }
        //            }
        //        };
        //    }
        //}
    }
}
