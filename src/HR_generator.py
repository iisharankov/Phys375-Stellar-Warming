import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

class FindFluxes:
    """
    Designed to overlay a ds9 reg file on top of a fits file of the same
    name and calculate the total flux of the area in the circle for every
    region in the reg file. Outputs a text file of circles and their sums
    """
    exposure_combos = []

    def __init__(self,  fit_file, reg, calibration_folder=False):
        """
        :param fit_file: string - name of fits file to process
        :param reg: string - name of reg file containing regions
        :param calibration_folder: bool - Bool if a calibration folder with valid files exits
        """
        self.output_data = []  # Float: Processed sums corresponding to every circle
        self.list_of_points = []  # Tuple of floats: xcord, ycord, and radius of each region
        self.list_of_calibrated_points = []  # Tuple of floats: RA, Dec, and arc_rad of each region

        # Next are used in processing of each file, initialized here
        self.data = self.mask = self.aperture = self.hdu = []
        self.total_sum = self.weighted_data = []

        self.cal_folder = calibration_folder  # bool to tell if cal folder exists

        # See __init__ for description
        self.fits = fit_file
        self.reg = reg
        self.cal_reg = (CAL_FOLDER + reg.split("/")[-1])

        # Extracts .fits metadata from header
        hdu_list = fits.open(self.fits)
        self.date_time = hdu_list[0].header['DATE-OBS'].replace("T", " ")
        self.exposure = int(np.floor(float(hdu_list[0].header['EXPTIME'])))
        self.filter = hdu_list[0].header['FILTER']

    def open_reg(self, reg_file, calibrated):
        """
        Opens the registry file, deletes header, and stores as variable
        :return: N/A
        """

        with open(reg_file) as temp_reg_file:

            for line in temp_reg_file:
                if line.startswith("circle"):  # if line not header

                    _temp_list = line[7:].split(")")  # separate values and meta info
                    values = _temp_list[0].split(",")  # sep values by commas

                    if calibrated:
                        self.list_of_calibrated_points.append(
                            (values[0], values[1], values[2],
                            _temp_list[1]))  # meta data

                    else:
                        self.list_of_points.append(
                            (np.round(float(values[0]), 6),
                            np.round(float(values[1]), 6),
                            np.round(float(values[2]), 6),
                            _temp_list[1]))  # meta data

    # @staticmethod
    def output_to_file(self, name, list_of_points, list_of_calibrated_points, list_of_sums):
        """
        Takes processed data and saves a text file
        :param name: str - name of input filename
        :param list_of_points: list of tuples of floats - (xcord, ycord, radius)
        :param list_of_calibrated_points: list of tuples of str - (Ra, Dec, arcrad)
        :param list_of_sums: list of floats - sums of pixels in region
        :return: N/A
        """

        with open(name + f"_DATA_{self.filter}{self.exposure}.txt", "w") as temp_text_file:
            if  self.exposure not in self.exposure_combos:
                self.exposure_combos.append(self.exposure)
            # Writes a header for metadata needed later
            temp_text_file.write(f"Header: date-time: \t {self.date_time} \t "
                                 f"Exposure: \t {self.exposure} \t Filter: \t {self.filter} \n ")
            temp_text_file.write("Header: \t x-axis \t y-axis \t radius \t sum \t error \t mag (-2.5log(sum)) \t "
                                 "az \t alt \t zen \n")  # Add a simple header

            # Print every region onto a new line
            for i, sum, cal_i in zip(list_of_points, list_of_sums, list_of_calibrated_points):
                az, alt = self.ra_dec_to_alt_az(cal_i[0], cal_i[1])

                # Found online, error in sum is sqrt of sum
                # (https://cxc.harvard.edu/ciao/ahelp/dmextract.html)
                err_in_sum = np.round(np.sqrt(sum), 6)

                # Output of line to text file, in header format shown above
                temp_text_file.write(f"{i[0]} \t {i[1]} \t {i[2]} \t "
                                     f"{sum} \t {err_in_sum} \t {np.round(-2.5*np.log10(sum), 6)} \t"
                                     f"{az} \t {alt} \t {np.round(90-float(alt), 6)}\n")

    def ra_dec_to_alt_az(self, region_ra, region_dec):
        """
        Given an RA and Dec, return the Az, Alt of the region
        :param region_ra: str - RA in hourangle format
        :param region_dec: str - Dec in hourangle format
        :return: tuple of str - Az, Alt in string form
        """

        # Make a SkyCoord obj with the Ra and Dec of the given region
        c = SkyCoord(ra=region_ra, dec=region_dec, unit=(u.hourangle, u.deg), frame='fk5')

        # Set the location to Waterloo, Ontario
        loc = EarthLocation(lat=43.4643, lon=-80.5204, height=340 * u.m)
        time = Time(self.date_time)  # Set time to night of observation

        # Calculate the Alt/Az
        # This line takes about 90% of the comp time, but it's due to
        # the overhead in astropy, so I can't really avoid it
        cAltAz = c.transform_to(AltAz(obstime=time, location=loc))

        return cAltAz.to_string(style='decimal').split(" ")

    def find_sum(self, entry):
        """
        Given a single entry (i.e region (i.e circle with radius and position)), this
        method creates a CirclePixelRegion obj of the identical radius and position
        as the region, makes a mask on that region from 0 to 1, and sums the weighted
        individual values of every pixel in the masked region.
        :param entry: tuple of floats - (xcord, ycord, radius)
        :return: N/A
        """
        xcord, ycord, radius, _other = entry  # 'other' includes region metadata

        # Creates the circle as a PixelRegion obj
        self.aperture = CirclePixelRegion(PixCoord(xcord, ycord), radius)

        # mask -> array of same size between of 0 to 1. If in mask, 1, otherwise 0
        self.mask = self.aperture.to_mask(mode='exact')  # "in" means within aperture

        self.hdu = fits.open(self.fits)[0]  # open fits file header
        self.data = self.mask.cutout(self.hdu.data)  # cuts out array not within mask

        # Weighted array (0 to 1) of % of pixel in mask
        self.weighted_data = self.mask.multiply(self.hdu.data)

        self.total_sum = np.round((np.sum(self.weighted_data)), 1)
        self.output_data.append(self.total_sum)

    def main(self):
        """
        main method to use as a call to start the process
        :return: N/A
        """

        # self.output_data = []
        self.open_reg(self.reg, False)  # opens uncalibrated files to list
        if self.cal_folder:  # opens calibrated files and adds to a list
            self.open_reg(self.cal_reg, True)

        for point in self.list_of_points:
            self.find_sum(point)

        self.output_to_file(self.fits[:-4],
                            self.list_of_points,
                            self.list_of_calibrated_points,
                            self.output_data)

    def find_exposure_combos(self):
        """
        Finds all exposures text files, and pairs them up.
        :return:
        """

        text_files = []
        for i in self.exposure_combos:
            pair = glob.glob(FOLDER + f"*{i}.txt")

            if not text_files:
                text_files.append(pair)
            else:
                for i in text_files:
                    if not set(pair) == set(i):
                        text_files.append(pair)
                        break
        self.colour_mags_to_txt(text_files)

    @staticmethod
    def find_calibrations(b, v, zen_b, zen_v):
        V = v -  (1 / np.cos(zen_v)) + (b - v)
        B = b -  (1 / np.cos(zen_b)) + (b - v)
        return B, V

    def b_v_finder(self, file1, file2):
        """
        Given two files, prints an output that is the sum divisor of the two
        :param file1: str - first text file
        :param file2: str - second text file
        :return:
        """
        self.colour_mags = []

        with open(file1) as b_file, open(file2) as v_file:

            for b_line, v_line in zip(b_file, v_file):
                if "Exposure:" in b_line:
                    exposure = str(b_line.split("\t")[3]).replace(" ", "")

                if "Header" not in b_line:
                    b_line = b_line.split("\t")
                    v_line = v_line.split("\t")
                    v_sum = float(v_line[3])
                    b_sum = float(b_line[3])
                    v_sum_error = float(v_line[4])
                    b_sum_error = float(b_line[4])
                    b = float(b_line[5])
                    v = float(v_line[5])
                    zen_b = float(b_line[8])
                    zen_v = float(v_line[8])

                    # b_minus_v_error = np.sqrt((v_error/1)**2 + (b_error/1)**2)
                    B, V = self.find_calibrations(b, v, zen_b, zen_v)
                    self.colour_mags.append((B, V, B - V))

        return exposure


    def colour_mags_to_txt(self, text_files):

        for duel_files in text_files:
            exposure = self.b_v_finder(duel_files[0], duel_files[1])
            self.plot_it(exposure)

            with open(FOLDER + f"B-V_{exposure}s.txt", "w") as file:
                file.write("B \t V \t B-V \n")

                for line in self.colour_mags:
                    file.write(f"{np.round(line[0], 6)} \t "
                               f"{np.round(line[1], 6)} \t "
                               f"{np.round(line[2], 6)} \n")

    def plot_it(self, exposure):
        v_list = [item[1] for item in self.colour_mags]
        b_minus_v_list = [item[2] for item in self.colour_mags]

        # finds index of min value and pops it. it's an outlier.
        val, idx = max((val, idx) for (idx, val) in enumerate(b_minus_v_list))
        b_minus_v_list.pop(idx)
        v_list.pop(idx)

        plt.plot(b_minus_v_list, v_list, ".", c='black')
        plt.title(f"B-V index for {exposure} seconds")
        plt.xlabel("B-V")
        plt.ylabel("V")
        ax = plt.gca()
        ax.invert_yaxis()
        plt.savefig(f"B-V_for_{exposure}s.png")
        plt.show()

    def plot_the_star(self, xcord, ycord):
        """
        Depreciated. Simply plots the graph and shows the star and region
        in the fits file.

        :param xcord: float - circle region x cord
        :param ycord: float - circle region y cord
        :return: N/A
        """
        plt.subplot(2, 2, 1)
        plt.title("Mask", size=12)
        plt.imshow(self.mask.data, origin='lower', extent=self.mask.bbox.extent)

        plt.subplot(2, 2, 2)
        plt.title("Data cutout", size=12)
        plt.imshow(self.data, origin='lower', extent=self.mask.bbox.extent)

        plt.subplot(2, 2, 3)
        plt.title("Data cutout multiplied by mask", size=9)
        plt.imshow(self.weighted_data, origin='lower', extent=self.mask.bbox.extent)

        ax = plt.subplot(2, 2, 4)
        plt.title("Mask in surrounding area", size=9)
        ax.imshow(self.hdu.data, cmap=plt.cm.viridis,interpolation='nearest', origin='lower')
        ax.add_artist(self.mask.bbox.as_artist(facecolor='none', edgecolor='white'))
        ax.add_artist(self.aperture.as_artist(facecolor='none', edgecolor='orange'))
        ax.set_xlim(xcord-20, xcord+20)
        ax.set_ylim(ycord-20, ycord+20)
        plt.show()


def fit_and_reg_checker(folder):
    """
    Makes sure for every fits file in provided folder, a reg file of same name
    is given and exists in the same location. Necessary to distinguish each
    region file to its corresponding .fit file
    :param folder: str - folder location
    :return: list of str - names of all filenames in that directory
    """
    list_to_append = []

    # Renames extensions that are .fits to .fit
    for fits_ext in glob.glob(folder + "*.fits"):
        os.rename(fits_ext, fits_ext[:-4] + "fit")

    # Finds all the fits files in the specified folder
    for file in glob.glob(folder + "*.fit"):

        try:
            # splits extension out of filename
            extensionless_filename = (file.split(".")[0])
            # Makes sure each fits file has corresponding .reg file
            assert os.path.exists(extensionless_filename + '.reg')

        except AssertionError as E:
            print(f"A given fits file did not have a .reg file "
                  f"of the same name! Check your filenames")
            raise E

        # add said filename to list if no exception
        else:
            list_to_append.append(extensionless_filename)

    return list_to_append


def check_cal_folder():
    """
    Checks if 'calibration' folder exists, and if so, checks it contains same
    .fits and .reg  files in the exact same format as in the parent folder.
    Critical for adding RA/Dec & Az/Alt info to each region in the text files.
    :return: list of str - names of each file found in the global FOLDER var
    """

    # Non_cal_files is basically files in the FOLDER level
    non_cal_files = fit_and_reg_checker(FOLDER)

    # Checks if the "calibrated" folder exists in the FOLDER dir
    cal_exists = os.path.exists(CAL_FOLDER)

    if not cal_exists:
        print("No calibrated file found, won't find Ra/Dec and Alt/Az "
              "to add to text files, can't find B-V mags without this!")
    else:
        print("calibrated file found")

        # calibrated_files is basically files in the calibrated folder level
        calibrated_files = fit_and_reg_checker(CAL_FOLDER)

        # Checks that exactly the same filenames exist in the two locations, no more, no less.
        try:
            a = [i.split("/")[-1] for i in calibrated_files]
            b = [i.split("/")[-1] for i in non_cal_files]
            assert a == b

        except AssertionError as E:
            print("Calibrated folder must the exact same number of files, with all"
                  "the exact same name as those in its parent directory")
            print(a, b)
            raise E

    return cal_exists, non_cal_files


def main():
    cal_exists, list_of_files = check_cal_folder()

    for file_index in range(len(list_of_files)):
        flux_finder = FindFluxes(list_of_files[file_index] + ".fit",
                                 list_of_files[file_index] + ".reg", cal_exists)
        # flux_finder.new_file_init(list_of_files[file_index] + ".fit",
        #                           list_of_files[file_index] + ".reg", cal_exists)
        flux_finder.main()
        print(f"Done file {file_index+1} of {len(list_of_files)}")

    flux_finder.find_exposure_combos()


if __name__ == '__main__':
    FOLDER = "Data/"
    CAL_FOLDER = FOLDER + 'calibrated/'

    main()
