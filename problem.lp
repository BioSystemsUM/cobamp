\ENCODING=ISO-8859-1
\Problem name: tmp_2AC3IXOK8FX0/7NNV01Y1O.lp

Minimize
 obj: 0 C + irr0_ind + irr1_ind + irr2_ind + irr3_ind + irr4_ind + irr5_ind
      + irr6_ind + irr7_ind + revfw0_ind + revbw0_ind
Subject To
 C_0#0:                      irr0 - irr1 - irr3 - irr5  = 0
 C_1#1:                      irr1 - irr2  = 0
 C_2#2:                      irr1 - irr3 + revfw0 - revbw0  = 0
 C_3#3:                      - irr5 + irr4  = 0
 C_4#4:                      irr5 - irr6  = 0
 C_5#5:                      irr3 + irr6 - irr7  = 0
 Cirr0_ind_helper0#6:        irr0_ind + irr0_inda  = 1
 Cirr0_ind_helper1#7:        - irr0_inda + irr0_indb >= 0
 Cirr0_ind_helper2#8:        - irr0_ind + irr0_indc  = 0
 Cirr0_ind_helper3#9:        - irr0_indc + irr0_indd >= 0
 Cirr1_ind_helper0#10:       irr1_ind + irr1_inda  = 1
 Cirr1_ind_helper1#11:       - irr1_inda + irr1_indb >= 0
 Cirr1_ind_helper2#12:       - irr1_ind + irr1_indc  = 0
 Cirr1_ind_helper3#13:       - irr1_indc + irr1_indd >= 0
 Cirr2_ind_helper0#14:       irr2_ind + irr2_inda  = 1
 Cirr2_ind_helper1#15:       - irr2_inda + irr2_indb >= 0
 Cirr2_ind_helper2#16:       - irr2_ind + irr2_indc  = 0
 Cirr2_ind_helper3#17:       - irr2_indc + irr2_indd >= 0
 Cirr3_ind_helper0#18:       irr3_ind + irr3_inda  = 1
 Cirr3_ind_helper1#19:       - irr3_inda + irr3_indb >= 0
 Cirr3_ind_helper2#20:       - irr3_ind + irr3_indc  = 0
 Cirr3_ind_helper3#21:       - irr3_indc + irr3_indd >= 0
 Cirr4_ind_helper0#22:       irr4_ind + irr4_inda  = 1
 Cirr4_ind_helper1#23:       - irr4_inda + irr4_indb >= 0
 Cirr4_ind_helper2#24:       - irr4_ind + irr4_indc  = 0
 Cirr4_ind_helper3#25:       - irr4_indc + irr4_indd >= 0
 Cirr5_ind_helper0#26:       irr5_ind + irr5_inda  = 1
 Cirr5_ind_helper1#27:       - irr5_inda + irr5_indb >= 0
 Cirr5_ind_helper2#28:       - irr5_ind + irr5_indc  = 0
 Cirr5_ind_helper3#29:       - irr5_indc + irr5_indd >= 0
 Cirr6_ind_helper0#30:       irr6_ind + irr6_inda  = 1
 Cirr6_ind_helper1#31:       - irr6_inda + irr6_indb >= 0
 Cirr6_ind_helper2#32:       - irr6_ind + irr6_indc  = 0
 Cirr6_ind_helper3#33:       - irr6_indc + irr6_indd >= 0
 Cirr7_ind_helper0#34:       irr7_ind + irr7_inda  = 1
 Cirr7_ind_helper1#35:       - irr7_inda + irr7_indb >= 0
 Cirr7_ind_helper2#36:       - irr7_ind + irr7_indc  = 0
 Cirr7_ind_helper3#37:       - irr7_indc + irr7_indd >= 0
 Crevfw0_ind_helper0#38:     revfw0_ind + revfw0_inda  = 1
 Crevfw0_ind_helper1#39:     - revfw0_inda + revfw0_indb >= 0
 Crevfw0_ind_helper2#40:     - revfw0_ind + revfw0_indc  = 0
 Crevfw0_ind_helper3#41:     - revfw0_indc + revfw0_indd >= 0
 Crevbw0_ind_helper0#42:     revbw0_ind + revbw0_inda  = 1
 Crevbw0_ind_helper1#43:     - revbw0_inda + revbw0_indb >= 0
 Crevbw0_ind_helper2#44:     - revbw0_ind + revbw0_indc  = 0
 Crevbw0_ind_helper3#45:     - revbw0_indc + revbw0_indd >= 0
 _E0#46:                     revfw0_ind + revbw0_ind <= 1
 KShortestSizeConstraint#47: irr0_ind + irr1_ind + irr2_ind + irr3_ind
                             + irr4_ind + irr5_ind + irr6_ind + irr7_ind
                             + revfw0_ind + revbw0_ind  = 8
 cut0#48:                    irr0_ind + irr3_ind + irr7_ind + revfw0_ind
                             + revbw0_ind <= 3
 cut0#49:                    irr0_ind + irr1_ind + irr2_ind + revfw0_ind
                             + revbw0_ind <= 3
 cut0#50:                    irr0_ind + irr1_ind + irr2_ind + irr3_ind
                             + irr7_ind <= 4
 cut0#51:                    irr0_ind + irr4_ind + irr5_ind + irr6_ind
                             + irr7_ind <= 4
 Cirr0_ind_ind1:   irr0_indb = 1 -> irr0  = 0
 Cirr0_ind_ind2:   irr0_indd = 1 -> irr0 - C >= 0
 Cirr1_ind_ind1:   irr1_indb = 1 -> irr1  = 0
 Cirr1_ind_ind2:   irr1_indd = 1 -> irr1 - C >= 0
 Cirr2_ind_ind1:   irr2_indb = 1 -> irr2  = 0
 Cirr2_ind_ind2:   irr2_indd = 1 -> irr2 - C >= 0
 Cirr3_ind_ind1:   irr3_indb = 1 -> irr3  = 0
 Cirr3_ind_ind2:   irr3_indd = 1 -> irr3 - C >= 0
 Cirr4_ind_ind1:   irr4_indb = 1 -> irr4  = 0
 Cirr4_ind_ind2:   irr4_indd = 1 -> irr4 - C >= 0
 Cirr5_ind_ind1:   irr5_indb = 1 -> irr5  = 0
 Cirr5_ind_ind2:   irr5_indd = 1 -> irr5 - C >= 0
 Cirr6_ind_ind1:   irr6_indb = 1 -> irr6  = 0
 Cirr6_ind_ind2:   irr6_indd = 1 -> irr6 - C >= 0
 Cirr7_ind_ind1:   irr7_indb = 1 -> irr7  = 0
 Cirr7_ind_ind2:   irr7_indd = 1 -> irr7 - C >= 0
 Crevfw0_ind_ind1: revfw0_indb = 1 -> revfw0  = 0
 Crevfw0_ind_ind2: revfw0_indd = 1 -> revfw0 - C >= 0
 Crevbw0_ind_ind1: revbw0_indb = 1 -> revbw0  = 0
 Crevbw0_ind_ind2: revbw0_indd = 1 -> revbw0 - C >= 0
Bounds
      C >= 1
 0 <= irr0_ind <= 1
 0 <= irr1_ind <= 1
 0 <= irr2_ind <= 1
 0 <= irr3_ind <= 1
 0 <= irr4_ind <= 1
 0 <= irr5_ind <= 1
 0 <= irr6_ind <= 1
 0 <= irr7_ind <= 1
 0 <= revfw0_ind <= 1
 0 <= revbw0_ind <= 1
 0 <= irr0_inda <= 1
 0 <= irr0_indb <= 1
 0 <= irr0_indc <= 1
 0 <= irr0_indd <= 1
 0 <= irr1_inda <= 1
 0 <= irr1_indb <= 1
 0 <= irr1_indc <= 1
 0 <= irr1_indd <= 1
 0 <= irr2_inda <= 1
 0 <= irr2_indb <= 1
 0 <= irr2_indc <= 1
 0 <= irr2_indd <= 1
 0 <= irr3_inda <= 1
 0 <= irr3_indb <= 1
 0 <= irr3_indc <= 1
 0 <= irr3_indd <= 1
 0 <= irr4_inda <= 1
 0 <= irr4_indb <= 1
 0 <= irr4_indc <= 1
 0 <= irr4_indd <= 1
 0 <= irr5_inda <= 1
 0 <= irr5_indb <= 1
 0 <= irr5_indc <= 1
 0 <= irr5_indd <= 1
 0 <= irr6_inda <= 1
 0 <= irr6_indb <= 1
 0 <= irr6_indc <= 1
 0 <= irr6_indd <= 1
 0 <= irr7_inda <= 1
 0 <= irr7_indb <= 1
 0 <= irr7_indc <= 1
 0 <= irr7_indd <= 1
 0 <= revfw0_inda <= 1
 0 <= revfw0_indb <= 1
 0 <= revfw0_indc <= 1
 0 <= revfw0_indd <= 1
 0 <= revbw0_inda <= 1
 0 <= revbw0_indb <= 1
 0 <= revbw0_indc <= 1
 0 <= revbw0_indd <= 1
Binaries
 irr0_ind  irr1_ind  irr2_ind  irr3_ind  irr4_ind  irr5_ind  irr6_ind 
 irr7_ind  revfw0_ind  revbw0_ind  irr0_inda  irr0_indb  irr0_indc  irr0_indd 
 irr1_inda  irr1_indb  irr1_indc  irr1_indd  irr2_inda  irr2_indb  irr2_indc 
 irr2_indd  irr3_inda  irr3_indb  irr3_indc  irr3_indd  irr4_inda  irr4_indb 
 irr4_indc  irr4_indd  irr5_inda  irr5_indb  irr5_indc  irr5_indd  irr6_inda 
 irr6_indb  irr6_indc  irr6_indd  irr7_inda  irr7_indb  irr7_indc  irr7_indd 
 revfw0_inda  revfw0_indb  revfw0_indc  revfw0_indd  revbw0_inda  revbw0_indb 
 revbw0_indc  revbw0_indd 
End
