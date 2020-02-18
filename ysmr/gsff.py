#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Copyright 2019, 2020 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de) and Jerôme
https://github.com/schwanbeck/YSMR
Original Idea and Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
(DOI: 10.1007/s12555-018-0938-4)
Many thanks to Prof. Pak, who kindly provided his original Matlab-Code for the Gaussian Sum FIR Filter.

##See also:
https://github.com/schwanbeck/GaussianSumFIRFilter

##Explanation
This file contains the Gaussian Sum Finite Impulse Response Filter used by YSMR.
This file is part of YSMR. YSMR is free software: you can distribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. YSMR is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with YSMR. If
not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


class GaussianSumFIR:
    """Algorithm 1 from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
    (DOI: 10.1007/s12555-018-0938-4) with changes by Jerôme Dretzke and Julian Schwanbeck
    """

    def __init__(self, delta_t, n_min=0, n_max=30, n_f=3,
                 a=None, c=None, likelihood_minimum=10 ** -20,
                 inv_cov=np.linalg.inv(np.eye(2)), x_hat_array_length=2):
        """Initialise the Gaussian sum finite impulse response filter

        Default for a:
        a = np.array([  # 4x4
             [1, 0, delta_time, 0, ],
             [0, 1, 0, delta_time, ],
             [0, 0, 1, 0, ],
             [0, 0, 0, 1, ],
        ], dtype=np.float)  # Equation (4)

        default for c:
        c = np.array([  # 2x4
            [1, 0, 0, 0, ],
            [0, 1, 0, 0, ],
        ])  # Equation (6)

        Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
        (DOI: 10.1007/s12555-018-0938-4)

        :param delta_t: time delta
        :type delta_t: float
        :param n_min: lower end of window size minus step
        :type n_min: int
        :param n_max: upper end of window size minus step
        :type n_max: int
        :param n_f: number of steps
        :type n_f: int
        :param a: Matrix A in Equation (4)
        :type a: np.array
        :param c: Matrix C in Equation (6)
        :type c: np.array
        :param likelihood_minimum: minimum for likelihood of filter
        :type likelihood_minimum: float
        :param inv_cov: inverted covariance matrix
        :type inv_cov: np.array
        :param x_hat_array_length: length for x_hat_array, either 2 or 4 for standard settings
        :type x_hat_array_length: int
        """
        self.likelihood_minimum = likelihood_minimum
        self.x_hat_array_length = x_hat_array_length
        self.n_f = n_f
        self.n_i = self.generate_n_i(n_min=n_min, n_max=n_max, n_f=self.n_f)
        self.gains = [self.compute_lsf_gain(
            filter_size=i,
            a=a,
            c=c,
            delta_time=delta_t
        ) for i in self.n_i]
        self.inv_cov = inv_cov

    @staticmethod
    def generate_n_i(n_min=0, n_max=30, n_f=3):
        """Calculate the filter sizes for the given minimum, maximum, and number of steps.
        See Equation (17).

        Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
        (DOI: 10.1007/s12555-018-0938-4)

        :param n_min: minimum filter horizon size minus step size
        :type n_min: int
        :param n_max: maximal filter horizon size
        :type n_max: int
        :param n_f: number of steps
        :type n_f: int
        :return: list of horizon sizes
        :rtype: list
        """
        p = (n_max - n_min) / n_f  # step size
        # P = (30 - 0) / 3 = 10

        n_i = [int(n_min + p * i) for i in range(1, n_f + 1)]  # range() is upper limit exclusive
        # n_i = [n_min + 1 * p, n_min + 2 * p, n_min + 3 * p]
        # n_i = [10, 20, 30]
        return n_i

    @staticmethod
    def compute_lsf_gain(filter_size, delta_time, a=None, c=None):
        """ Compute the least square filter gain
        See Equation (14) and (13)

        Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
        (DOI: 10.1007/s12555-018-0938-4)

        :param filter_size: filter horizon size
        :type filter_size: int
        :param delta_time: time delta
        :type delta_time: float
        :param a: Matrix A in Equation (4)
        :type a: np.array
        :param c: Matrix C in Equation (6)
        :type c: np.array
        :return: filter gain
        """
        if a is None:
            a = np.array([
                [1, 0, delta_time, 0, ],
                [0, 1, 0, delta_time, ],
                [0, 0, 1, 0, ],
                [0, 0, 0, 1, ],
            ], dtype=np.float)  # (4)
        if c is None:
            c = np.array([  # 2x4
                [1, 0, 0, 0, ],
                [0, 1, 0, 0, ],
            ])  # (6)

        h_bar = c
        a_n = a
        for _ in range(0, filter_size - 1):  # Equation (14)
            h_bar = np.concatenate((h_bar, np.dot(c, a_n)), axis=0)
            a_n = np.dot(a_n, a)
        l_bar = np.dot(h_bar, np.linalg.matrix_power(np.linalg.inv(a), filter_size))
        # Equation (13)
        lsf_gain = np.dot(
            np.linalg.inv(
                np.dot(l_bar.T, l_bar)), l_bar.T
        )
        return lsf_gain

    @staticmethod
    def lsff_calc(horizon_size, filter_gain, measurements):
        """Calculate prediction of lsf
        See Equation (12)

        Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
        (DOI: 10.1007/s12555-018-0938-4)

        :param horizon_size: Current filter horizon size
        :type horizon_size: int
        :param filter_gain: Filter gain as calculated by compute_lsf_gain()
        :param measurements: Current measurements
        :return: (4, 1) np.array
        :rtype: np.array
        """
        y_mat = measurements[-horizon_size:]

        # linearised to filter_size x 1
        y_mat = [cur_item for cur_tuple in y_mat for cur_item in cur_tuple]
        # y_mat = y_mat.reshape(-1)  # the same but somewhat slower

        # gain_i * y_mat -> (4, 20) * (20, 1) -> (4, 1) at filter size of 10
        return np.dot(filter_gain, y_mat)

    def likelihood_calc(self, measurement, y_hat):
        """Calculate likelihood for current filter
        See Equation (20)

        Equations from Pak, JM: Gaussian Sum FIR Filtering for 2D Target Tracking
        (DOI: 10.1007/s12555-018-0938-4)

        :param measurement: current measurement
        :param y_hat: prediction for current measurement
        :return: likelihood
        :rtype: float
        """
        y_y_hat_diff = measurement - y_hat
        try:  # Equation (20)
            likelihood = np.exp(
                -0.5 * np.dot(
                    y_y_hat_diff.T, np.dot(self.inv_cov, y_y_hat_diff)
                )
            )
        except FloatingPointError:  # happens at distances > ~10
            return self.likelihood_minimum
        if likelihood < self.likelihood_minimum:
            return self.likelihood_minimum
        return likelihood

    def predict(self, mode=0, previous_measurements=None, weight_array=None, x_hat_array=None, **kwargs):
        """Predict result based on previous weights.

        ## Example

        GaussianSumFIR.predict() takes as arguments the current dictionary of settings.
        GaussianSumFIR.predict() uses the settings as returned by GaussianSumFIR.correct().
        For initial setup, use:

        settings_dict = {}

        filtered_result, settings_dict = GaussianSumFIR.correct(measurement, **settings_dict)

        prediction, settings_dict = GaussianSumFIR.predict(**settings_dict)

        :param mode: current used filters
        :type mode: int
        :param previous_measurements: list of previous measurements
        :param weight_array: weights for lsf
        :param x_hat_array: array for results of lsf
        :param kwargs: dict of current key word arguments
        :type kwargs: dict
        :return: predicted result, kwargs
        """
        if previous_measurements is None:
            return None, kwargs
        for filter_index in range(0, mode):
            horizon = self.n_i[filter_index]
            gain = self.gains[filter_index]

            fir_x_hat = self.lsff_calc(
                horizon_size=horizon,
                filter_gain=gain,
                measurements=previous_measurements
            )

            x_hat_array[:, filter_index] = fir_x_hat[:self.x_hat_array_length]

        predict_x_hat = np.sum(x_hat_array * weight_array, axis=1)  # predict using old weights
        kwargs.update({
            'mode': mode,
            'previous_measurements': previous_measurements,
            'weight_array': weight_array,
            'x_hat_array': x_hat_array,
        })
        return predict_x_hat, kwargs

    def correct(self, measurement, mode=0, previous_measurements=None, weight_array=None,
                likelihood_array=None, x_hat_array=None, **kwargs):
        """Correct signal based on prediction and current input.

        ## Example

        GaussianSumFIR.correct() takes as arguments the current measurement
        and a dictionary of settings.
        GaussianSumFIR.correct() creates it's own settings upon first use.
        For initial setup, use:

        settings_dict = {}  # initialise with empty dict

        filtered_result, settings_dict = GaussianSumFIR.correct(measurement, **settings_dict)

        prediction, settings_dict = GaussianSumFIR.predict(**settings_dict)

        :param measurement: new measurement
        :param mode: current used filters
        :type mode: int
        :param previous_measurements: list of previous measurements
        :param weight_array: weights for least square filter
        :param likelihood_array: likelihoods for each least square filter
        :param x_hat_array: array for results of least square filter
        :param kwargs: dict of current key word arguments
        :type kwargs: dict
        :return: filtered result, kwargs
        """
        if previous_measurements is None:
            # Initialise with first measurement
            previous_measurements = [measurement] * self.n_i[0]

        new_mode = False
        if mode < self.n_f:  # stop at max
            while len(previous_measurements) >= self.n_i[mode]:
                mode += 1
                new_mode = True
                if mode >= self.n_f:
                    break

        if new_mode:
            likelihood_array = [self.likelihood_minimum] * mode  # initialise with minimum value
            # 4 because of H_Ni / we just need 2 though
            x_hat_array = np.zeros((self.x_hat_array_length, mode))
            weight_array = 1 / mode * np.ones(mode)
            # call predict to update x_hat_array
            kwargs.update({
                'mode': mode,
                'previous_measurements': previous_measurements,
                'weight_array': weight_array,
                'likelihood_array': likelihood_array,
                'x_hat_array': x_hat_array,
            })

            _, kwargs_curr = self.predict(
                **kwargs
            )
            kwargs.update(kwargs_curr)

        for filter_index in range(0, mode):
            fir_x_hat = x_hat_array[:, filter_index]
            likelihood = self.likelihood_calc(measurement=measurement, y_hat=fir_x_hat[:2], )
            likelihood_array[filter_index] = likelihood

        previous_measurements.append(measurement)
        # Remove unnecessary measurements
        if len(previous_measurements) > self.n_i[-1] + 1:
            previous_measurements = previous_measurements[-(self.n_i[-1] + 1):]

        try:  # mostly works
            weight_sum = sum(likelihood_array * weight_array)
        except FloatingPointError:  # unless it doesn't
            weight_sum = 0
            for i in range(len(likelihood_array)):
                try:
                    weight_sum += likelihood_array[i] * weight_array[i]
                except FloatingPointError:
                    weight_sum += self.likelihood_minimum
                    weight_array[i] = self.likelihood_minimum

        # Calculate new weights
        for filter_index in range(0, weight_array.shape[0]):
            likeli_curr = likelihood_array[filter_index] * weight_array[filter_index] / weight_sum
            weight_array[filter_index] = likeli_curr

        # Output
        x_hat = np.sum(x_hat_array * weight_array, axis=1)

        # Return Prediction, settings
        kwargs.update({
            'mode': mode,
            'previous_measurements': previous_measurements,
            'weight_array': weight_array,
            'likelihood_array': likelihood_array,
            'x_hat_array': x_hat_array,
        })
        return x_hat, kwargs
