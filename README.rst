.. raw:: html

   <div style="display: flex; align-items: center;">
     <img src="https://github.com/SouthPoleTelescope/spt_candl_data/blob/main/logos/spt_logo.jpg?raw=true" style="width:35%;"/>
     <img src="https://github.com/SouthPoleTelescope/spt_candl_data/blob/main/logos/blank_padding.png?raw=true" style="width:5%;"/>
     <img src="https://github.com/Lbalkenhol/candl/raw/main/docs/logos/candl_wordmark&symbol_col_RGB.png" style="width:55%;"/>
   </div>
   
   <h1>Official SPT data for <tt>candl</tt></h1>

Official SPT data for the differentiable CMB likelihood framework `candl <https://github.com/Lbalkenhol/candl>`_.

Installation
------------

To install the SPT candl data library, simply navigate to where you would like to store the data and then run::

    git clone https://github.com/SouthPoleTelescope/spt_candl_data.git
    cd spt_candl_data
    pip install .

This will download the relevant data files. The installation gives you access to handy short cuts that make it easier to initialise the likelihoods.
Note that you also need to install `candl <https://github.com/Lbalkenhol/candl>`_ in order to run the likelihoods, run ``pip install candl`` or see the repository for more detailed instructions.

To check that everything is working, you can test the supplied likelihoods (provided you have already installed candl) via::

    import spt_candl_data
    spt_candl_data.run_all_tests()

Available Data
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 10

   * - Name
     - Key Shortcuts
     - References
     - ``candl`` version used for publication

   * - SPT-3G D1 T&E
     - ``spt_candl_data.SPT3G_D1_TnE``, ``spt_candl_data.SPT3G_D1_TnE_lite``
     - | `Camphuis et al. 2025 <https://pole.uchicago.edu/public/Home.html>`__
      
       Quan et al. 2025 (in prep.)
     - ``2.0.3``

You can also get a detailed summary of the variants of all likelihoods available, by running the following python code::

    import spt_candl_data
    spt_candl_data.print_all_shortcuts()

Additional Info
^^^^^^^^^^^^^^^^^^

For subsets of the SPT-3G D1 T&E likelihood, such as fitting only EE spectra, use the corresponding ``variant`` of the likelihood. Avoid using the ``candl`` keyword ``data_selection``, unless for restricting the angular scales used.

Getting Started
--------------

We supply files to help you use SPT data with common cosmological samplers as well as tutorials on how to interact with the data and perform common analysis tasks.
You can find more help and tutorials in the `candl documentation <http://candl.readthedocs.io>`_.

Notebooks
^^^^^^^^^^^^^^

``tutorial_notebooks/SPT3G_D1_TnE_tutorial.ipynb``: this notebook uses the SPT-3G D1 T&E likelihoods and is split into two parts. The first one shows you how to initialize the multifrequency likelihood, evaluate it, visualize the data, and helps you understand the data model. The second part uses the `SPTlite` likelihood and leverages the differentiability of `candl`. This part shows how to translate biases from the band-power level to the parameter-level and how to perform gradient-based minimization and sampling.

Cobaya
^^^^^^^^^^^^^^

You can find template Cobaya ``.yaml`` files to help you launch chains as well as ΛCDM proposal matrices in the ``cobaya/`` folder

MontePython
^^^^^^^^^^^^^^

Montepython likelihood folders you can copy-paste into your MontePython installation are in the ``montepython/`` folder.
For each likelihood, there is a template ``.param`` file and a ΛCDM proposal matrix.
Note that the MontePython likelihoods by default do not remove the ``candl`` internal priors. For more information, see the `candl documentation <http://candl.readthedocs.io>`_.

CosmoSIS
^^^^^^^^^^^^^^

You can find template CosmoSIS ``.ini`` files to help you launch chains in the ``cosmosis/`` folder.

How to cite this data
------------------------

Please cite:

* the paper(s) of the relevant likelihood(s) you use (see the data summary table above for quick reference) and
* the `candl release paper <https://arxiv.org/abs/2401.13433>`_.

===================

.. |NSF| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/NSF.jpg
   :alt: NSF
   :height: 150px

.. |USAP| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/USAP.jpg
   :alt: USAP
   :height: 150px

.. |DOE| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/DOE.jpg
   :alt: DOE
   :height: 150px

.. |KICP| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/KICP.png
   :alt: KICP
   :height: 150px

.. |ERC| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/ERC.jpg
   :alt: ERC
   :height: 150px

.. |neucosmos| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/neucosmos_logo.png
   :alt: NEUCosmoS
   :height: 125px

.. |sorbonne| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/sorbonne_logo.jpeg
   :alt: Sorbonne
   :height: 100px

.. |IAP| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/IAP_logo.jpeg
   :alt: IAP
   :height: 150px

.. |cnrs| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/cnrs_logo.jpeg
   :alt: CNRS
   :height: 150px

.. |argonne| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/argonne.jpg
   :alt: Argonne
   :height: 100px

.. |fermilab| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/fermilab.jpg
   :alt: Fermilab
   :height: 80px

.. |case_western| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/case_western.jpg
   :alt: Case Western
   :height: 100px

.. |mcgill| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/mcgill.jpg
   :alt: McGill
   :height: 100px

.. |melb| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/melb.jpg
   :alt: Melbourne
   :height: 100px

.. |michigan| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/michigan.jpg
   :alt: Michigan
   :height: 100px

.. |SLAC| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/SLAC.jpg
   :alt: SLAC
   :height: 80px

.. |stanford| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/stanford.jpg
   :alt: Stanford
   :height: 125px

.. |berkeley| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/berkeley.jpg
   :alt: Berkeley
   :height: 80px

.. |davis| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/davis.jpg
   :alt: Davis
   :height: 100px

.. |chicago| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/chicago.jpg
   :alt: Chicago
   :height: 100px

.. |boulder| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/boulder.jpg
   :alt: Boulder
   :height: 100px

.. |uoi| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/uoi.jpg
   :alt: University of Illinois
   :height: 100px

.. |caps| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/caps.png
   :alt: CAPS
   :height: 100px

.. |harvard| image:: https://github.com/SouthPoleTelescope/spt_candl_data/raw/main/logos/sponsors_institutions/harvard.jpg
   :alt: Harvard
   :height: 100px

|NSF| |USAP| |DOE| |ERC| |cnrs|

|IAP| |neucosmos| |sorbonne|

|chicago| |davis| |mcgill|

|berkeley| |stanford| |SLAC|

|fermilab| |argonne|

|melb| |michigan| |case_western| 

|uoi| |caps|

|boulder| |harvard|
