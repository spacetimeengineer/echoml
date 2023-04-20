# coding=utf-8

#  THIS SOFTWARE AND ITS DOCUMENTATION ARE CONSIDERED TO BE IN THE PUBLIC DOMAIN
#  AND THUS ARE AVAILABLE FOR UNRESTRICTED PUBLIC USE. THEY ARE FURNISHED "AS
#  IS." THE AUTHORS, THE UNITED STATES GOVERNMENT, ITS INSTRUMENTALITIES,
#  OFFICERS, EMPLOYEES, AND AGENTS MAKE NO WARRANTY, EXPRESS OR IMPLIED,
#  AS TO THE USEFULNESS OF THE SOFTWARE AND DOCUMENTATION FOR ANY PURPOSE.
#  THEY ASSUME NO RESPONSIBILITY (1) FOR THE USE OF THE SOFTWARE AND
#  DOCUMENTATION; OR (2) TO PROVIDE TECHNICAL SUPPORT TO USERS.

"""
#####################################################################
# Developed by Mike Jech, michael.jech@noaa.gov
# 
# National Oceanic and Atmospheric Administration (NOAA)
# Northeast Fisheries Science Center (NEFSC)
# Woods Hole, MA USA
#
# utilities to work with echoPype data
#
# This code was modified from mask.py written by Rick Towler
#####################################################################
"""

import numpy as np
import matplotlib
import xarray as xr
import pandas as pd
#from echolab2.ping_data import ping_data


class seabedDetection():
    """
    Utilites to detect the seabed echo from echoPype data
    original code by Rick Towler
    modified by jech
    """
    
    def __init__(self, xrda): #, search_min=10, window_len=11, backstep=35):
        #self.search_min = search_min
        #self.window_len = window_len
        #self.backstep = backstep
        self.svdata = xrda
   
    def afscBotDetect(self, search_min=10, window_len=11, backstep=35):
        self.search_min = search_min
        self.window_len = window_len
        self.backstep = backstep

        v_axis = self.svdata.echo_range.values
        npings = len(self.svdata.ping_time)
        botline = np.empty(npings)
        botline[:] = np.nan

        if not np.any(v_axis > self.search_min):
        # there are no data beyond our minimum search range - nothing to do but return nan
            return botline

        iping = 0
  	    # iterate through each ping for bottom detection
        for pt in self.svdata.ping_time.values:
            # range slice
            vslice = self.svdata.loc[pt].values
            # skip pings that don't have at least some samples with data
            if (not np.all(np.isnan(vslice))):
                # determine the maximum Sv beyond the specified minimum range
                max_Sv = np.nanmax(vslice[v_axis > search_min])
                sample_max = np.nanargmax(vslice == max_Sv)
                # smooth ping
                hanning_window = np.hanning(self.window_len)
                smoothed_ping = np.convolve(hanning_window/hanning_window.sum(), 
                                            vslice, mode='same')
                # calculate the threshold that will define the lower bound 
                # (in Sv) of our echo envelope
                threshold = max_Sv - self.backstep
                # get the echo envelope
                botline[iping] = self.get_echo_envelope(smoothed_ping, sample_max,
		                                           threshold, v_axis, self.search_min,
		                                           contiguous=True)
            iping += 1
        # interpolate the nans
        botline = self.interp_nans(botline)
        return botline

    def get_echo_envelope(self, data, echo_peak, threshold, range_vector, 
                          range_min, contiguous=True):
        '''
        get_echo_envelope calculates the near and far edges of an echo defined by
        the provided peak sample value and a threshold value. You must also provide
        the ping vector, sample vector, and range vector.
        '''
        #  calculate the lower search bound - usually you will at least want to avoid the ringdown.
        lower_bound = np.nanargmax(range_vector > range_min)

        try:
            if lower_bound == echo_peak:
                min_range = 0
            else:
                #  then find the lower bound of the envelope
                near_envelope_samples = (echo_peak - 
                                         np.squeeze(np.where(data[echo_peak:lower_bound:-1] > 
                                                             threshold)))
                if contiguous:
                    sample_diff = np.where(np.diff(near_envelope_samples) < -1)
                    if (sample_diff[0].size > 0):
                        min_idx = np.min(sample_diff)
                        min_sample = near_envelope_samples[min_idx]
                    else:
                        min_sample = near_envelope_samples[-1]
                else:
                    min_sample = near_envelope_samples[-1]

                #  and the next one
                previous_sample = min_sample - 1
                #  calculate the interpolated range for our near envelope edge
                min_range = np.interp(threshold, [data[previous_sample], 
                                                  data[min_sample]],
                                                 [range_vector[previous_sample], 
                                                  range_vector[min_sample]])
        except:
            min_range = np.nan

        return min_range

    def interp_nans(self, y):
        """Linearly interpolate NaNs.
        
        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - input vector with lineraly interpolated nans
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            
        code from stackoverflow, eat & snake_charmer
        """
        nans,x = np.isnan(y), lambda z:z.nonzero()[0]
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

        return y
        

class lineMask():
    ''' Generate a mask based on lines. Examples include excluding data below
    the seabed detected echo, selecting data along a trawl path using headrope
    and footrope data
    '''
    
    def __init__(self, xrda):
        self.svdata = xrda
    
    def apply_between_line_segments(self, upper_line_segment, lower_line_segment, 
                                    value=True):
        """Sets mask elements between the provided line segment to the specified
        value. This does not require the line to span the entire echogram. See
        apply_line to do that. This does not interpolate! This only applies at 
        the nodes/vertices given in the line object.

        This is a convenience function. See apply_line_segment for details.

        Args:
            upper_line_segment (line): The line object used to define the upper 
                            boundary on the mask where the provided value will be
                            applied.
            lower_line_segment (line): The line object used to define the lower 
                            boundary on the mask where the provided value will be
                            applied.
            value (bool): Set this keyword to True to set the mask elements to
                          True, False to False. Default: True

        """

        self.apply_line_segment(upper_line_segment, value=value, 
                                other_line_segment=lower_line_segment)


    def apply_below_line_segment(self, line_segment, value=True):
        """Sets mask elements below the line segment to the specified value. 
        This does not require the line to span the entire echogram. See 
        apply_below_line to do that.
        This does not interpolate! This only applies at the 
        nodes/vertices given in the line object.

        This is a convenience function. See apply_line_segment for details.

        Args:
            line_obj (line): The line object used to define the upper boundary
                             on the mask where the provided value will be
                             applied. All mask elements at or below the line segment
                             will be set.
            value (bool): Set this keyword to True to set the mask elements to
                          True, False to False. Default: True

        """

        self.apply_line_segment(line_segment, value=value, apply_above=False)


    def apply_above_line_segment(self, line_segment, value=True):
        """Sets mask elements above the line segment to the specified value. 
        This does not require the line to span the entire echogram. See 
        apply_line_above to do that. This does not interpolate! This only 
        applies at the nodes/vertices given in the line object.

        This is a convenience function. See apply_line_segment for details.

        Args:
            line_obj (line): The line object used to define the lower boundary
                             on the mask where the provided value will be
                             applied. All mask elements at or above the line
                             will be set.
            value (bool): Set this keyword to True to set the mask elements to
                          True, False to False. Default: True

        """

        self.apply_line_segment(line_segment, value=value, apply_above=True)


    def apply_line_segment(self, line_segment, apply_above=False, value=True,
            other_line_segment=None):
        """Sets mask elements above, below, and between line segments. This does not
        require the line segment to span the entire echogram. See apply_line to 
        do that.
        This does not interpolate! This only applies at the nodes/vertices 
        given in the line object.

        This method sets this mask's elements above, below, or between the
        provided echolab2.processing.line object(s) to the specified boolean
        value.

        Set apply_above to True to apply the provided value to samples with
        range/depth values LESS THAN OR EQUAL TO the provided line segment.

        Set apply_above to False to apply the provided value to samples with
        range/depth values GREATER THAN OR EQUAL TO the provided line segment.

        If you set other_line to a line object, the apply_above argument will
        be ignored and samples greater than or equal to line_obj and samples
        less than or equal to other_line_segment will be set to the provided value.
        In other words, setting other_line_segment will set samples between the two
        line segments.


        Args:
            line_segment (line): The line used to define the vertical
                                 boundary for each
            apply_above (bool): Set apply_above to True to apply the provided
                                value to all samples equal to or less than the
                                line segment range/depth. Set to False to apply to
                                samples greater than or equal to the line segment 
                                range/depth. Default: False
            other_line (line): Set other_line_segment to a line to set all samples
                               between line and other_line_segment to the provided
                               value. Default: None
            value (bool): Set this keyword to True to set the mask elements to
                          True, False to False. Default: True

        """
        # Ensure value is a bool.
        value = bool(value)

        # get our vertical axis
        v_axis = self.svdata.echo_range.values

        npings = len(self.svdata.ping_time)

        # get the ping indices of the data object that match the line segments
        #didx = list(self.ping_time)
        #idx = [didx.index(min(didx, key=lambda t0: abs(t0-t1))) for t1 in line_obj.ping_time]

        #  first check if we're setting between two lines
        if other_line_segment is not None:
            # apply value to mask elements between the two provided line segments
            iping = 0
      	    # iterate through each ping for the line segment
            for pt in line_segment.ping_time.values:
                # range slice
                vslice = self.svdata.loc[pt].values
                samps_to_mask = v_axis >= line_segment.loc[pt].values
                samps_to_mask &= v_axis <= other_line_segment.loc[pt].values
                self.svdata.loc[pt][samps_to_mask] = value

        else:
            #  only one line passed so we'll apply above or below that line segment
            if apply_above:
                # apply value to mask elements less than or equal to the line segment
                for pt in line_segment.ping_time.values:
                    samps_to_mask = v_axis <= line_segment.loc[pt].values
                    self.svdata.loc[pt][samps_to_mask] = value
            else:
                # apply value to mask elements greater than or equal to the line
                for pt in line_segment.ping_time.values:
                    samps_to_mask = v_axis >= line_segment.loc[pt].values
                    self.svdata.loc[pt][samps_to_mask] = value


