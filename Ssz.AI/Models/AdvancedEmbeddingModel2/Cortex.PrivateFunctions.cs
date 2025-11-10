using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public partial class Cortex : ISerializableModelObject
{
    #region private functions

    public void CalculateActivityAndSuperActivity(Cortex.Memory cortexMemory, ActivitiyMaxInfo activitiyMaxInfo)
    {
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Data.Length,
            mci =>
            {
                var mc = MiniColumns.Data[mci];                
                mc.Temp_Activity = MiniColumnsActivityHelper.GetActivity(mc, cortexMemory.DiscreteRandomVector, Constants);
            });

        activitiyMaxInfo.MaxActivity = float.MinValue;
        activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

        //if (Constants.SuperactivityThreshold)
        //    activitiyMaxInfo.MaxSuperActivity = Constants.K4;
        //else
        activitiyMaxInfo.MaxSuperActivity = float.MinValue;
        activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

        foreach (var mc in MiniColumns.Data)
        {
            mc.Temp_SuperActivity = MiniColumnsActivityHelper.GetSuperActivity(mc, Constants);

            float a = mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity;
            if (a > activitiyMaxInfo.MaxActivity)
            {
                activitiyMaxInfo.MaxActivity = a;
                activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
            }
            else if (a == activitiyMaxInfo.MaxActivity)
            {
                activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
            }

            if (mc.Temp_SuperActivity > activitiyMaxInfo.MaxSuperActivity)
            {
                activitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
                activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
            }
            else if (mc.Temp_SuperActivity == activitiyMaxInfo.MaxSuperActivity)
            {
                activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
            }
        }
    }

    #endregion
}
