# DAN

**************************************************************************************************

Title   : "Stacking-based Deep Neural Network: Deep Analytic Network for Pattern Classification"

Authors : C. Y. Low*, J. Park, A. B. J. Teoh

Affl.   : Yonsei University, Seoul, South Korea; Multimedia University, Malaysia*

Email   : chengyawlow@gmail.com; julypraise@gmail.com; bjteoh@yonsei.ac.kr

DOI     : 10.1109/TCYB.2019.2908387, accepted by IEEE Transactions on Cybernetics, 

**************************************************************************************************

Deep Analytic Networks on Pre-trained VD-VGG-19 and ResNet-156 Features using TinyImageNet

1.  Download the pre-learned, pre-extracted VD-VGG-19 & ResNet-156 features

    (a) Pre-learned, pre-extracted VD-VGG-19 : https://drive.google.com/open?id=1yqflW8M_e5_tPxELu6HEPpyz3J1RWqBx
    
    (b) Pre-learned, pre-extracted ResNet-156 : https://drive.google.com/open?id=1N8w81BoV38Ub8fjQ_HPJAhIDpIqhXoAM
    
2.  Run DAN_TinyImageNet_Main

**************************************************************************************************

Top-1 Accuracy(%) for DAN + VD-VGG :

recogRate_FT_TT =

    L1 : 65.9800   
    L2 : 67.9500   
    L3 : 68.3400   
    L4 : 68.9800   
    L5 : 69.1800   
    L6 : 69.2600   
    L7 : 69.3000   
    L8 : 69.3400*

recogRate_BEST_FT =

    69.3400 ( 8 )

**************************************************************************************************

Top-1 Accuracy(%) for DAN + ResNet :

recogRate_FT_TT =

    L1 : 72.6900     
    L2 : 74.7700     
    L3 : 75.4400     
    L4 : 75.9800     
    L5 : 76.2800     
    L6 : 76.5900       
    L7 : 76.7500     
    L8 : 76.8400*

recogRate_BEST_FT =

    76.8400 ( 8 )

**************************************************************************************************

Top-1 Accuracy(%) for DAN + VD-VGG + ResNet :

recogRate_FT_TT =

    L1 : 74.9600   
    L2 : 77.1300
    L3 : 77.5300
    L4 : 77.8700 
    L5 : 78.0200*
    L6 : 77.9100
    L7 : 77.8300
    L8 : 77.6300

recogRate_BEST_FT =

    78.0200 ( 5 )

