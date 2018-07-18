\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 _4666cf9bm539am11e8m864ame0d55e08adc7#0: vp0_ind + vn0_ind + vp1_ind + vn1_ind
                                          + vp2_ind + vn2_ind + vp3_ind
                                          + vn3_ind + vp4_ind + vn4_ind
                                          + vp5_ind + vn5_ind + vp6_ind
                                          + vn6_ind + vp7_ind + vn7_ind
                                          + vp8_ind + vn8_ind
Subject To
 Ci0:             - vn0 + vp0 + u0 >= 0
 Ci1:             - u0 + vp1 - vn1 + u1 + u2 >= 0
 Ci2:             - u1 - vn2 + vp2 >= 0
 Ci3:             - u0 - u2 + u5 - vn4 + vp4 >= 0
 Ci4:             - vn5 + vp5 + u3 >= 0
 Ci5:             - u0 - u3 - vn6 + vp6 + u4 >= 0
 Ci6:             u5 - u4 + vp7 - vn7 >= 0
 Ci7:             - u5 + vp8 - w0 - vn8 >= 0
 Cr0:             u2 - vn3 + vp3  = 0
 Cb:              - w0 + C  = 0
 Act1_vp0_ind:    vp0_indc - vp0_ind  = 0
 Act2_vp0_ind:    - vp0_indc + vp0_indd >= 0
 Inact1_vp0_ind:  vp0_ind + vp0_inda  = 1
 Inact2_vp0_ind:  - vp0_inda + vp0_indb >= 0
 Act1_vn0_ind:    vn0_indc - vn0_ind  = 0
 Act2_vn0_ind:    - vn0_indc + vn0_indd >= 0
 Inact1_vn0_ind:  vn0_ind + vn0_inda  = 1
 Inact2_vn0_ind:  - vn0_inda + vn0_indb >= 0
 Act1_vp1_ind:    - vp1_ind + vp1_indc  = 0
 Act2_vp1_ind:    - vp1_indc + vp1_indd >= 0
 Inact1_vp1_ind:  vp1_ind + vp1_inda  = 1
 Inact2_vp1_ind:  - vp1_inda + vp1_indb >= 0
 Act1_vn1_ind:    - vn1_ind + vn1_indc  = 0
 Act2_vn1_ind:    - vn1_indc + vn1_indd >= 0
 Inact1_vn1_ind:  vn1_ind + vn1_inda  = 1
 Inact2_vn1_ind:  - vn1_inda + vn1_indb >= 0
 Act1_vp2_ind:    - vp2_ind + vp2_indc  = 0
 Act2_vp2_ind:    - vp2_indc + vp2_indd >= 0
 Inact1_vp2_ind:  vp2_ind + vp2_inda  = 1
 Inact2_vp2_ind:  - vp2_inda + vp2_indb >= 0
 Act1_vn2_ind:    - vn2_ind + vn2_indc  = 0
 Act2_vn2_ind:    - vn2_indc + vn2_indd >= 0
 Inact1_vn2_ind:  vn2_ind + vn2_inda  = 1
 Inact2_vn2_ind:  - vn2_inda + vn2_indb >= 0
 Act1_vp3_ind:    vp3_indc - vp3_ind  = 0
 Act2_vp3_ind:    - vp3_indc + vp3_indd >= 0
 Inact1_vp3_ind:  vp3_ind + vp3_inda  = 1
 Inact2_vp3_ind:  - vp3_inda + vp3_indb >= 0
 Act1_vn3_ind:    vn3_indc - vn3_ind  = 0
 Act2_vn3_ind:    - vn3_indc + vn3_indd >= 0
 Inact1_vn3_ind:  vn3_ind + vn3_inda  = 1
 Inact2_vn3_ind:  - vn3_inda + vn3_indb >= 0
 Act1_vp4_ind:    - vp4_ind + vp4_indc  = 0
 Act2_vp4_ind:    - vp4_indc + vp4_indd >= 0
 Inact1_vp4_ind:  vp4_ind + vp4_inda  = 1
 Inact2_vp4_ind:  - vp4_inda + vp4_indb >= 0
 Act1_vn4_ind:    - vn4_ind + vn4_indc  = 0
 Act2_vn4_ind:    - vn4_indc + vn4_indd >= 0
 Inact1_vn4_ind:  vn4_ind + vn4_inda  = 1
 Inact2_vn4_ind:  - vn4_inda + vn4_indb >= 0
 Act1_vp5_ind:    - vp5_ind + vp5_indc  = 0
 Act2_vp5_ind:    - vp5_indc + vp5_indd >= 0
 Inact1_vp5_ind:  vp5_ind + vp5_inda  = 1
 Inact2_vp5_ind:  - vp5_inda + vp5_indb >= 0
 Act1_vn5_ind:    - vn5_ind + vn5_indc  = 0
 Act2_vn5_ind:    - vn5_indc + vn5_indd >= 0
 Inact1_vn5_ind:  vn5_ind + vn5_inda  = 1
 Inact2_vn5_ind:  - vn5_inda + vn5_indb >= 0
 Act1_vp6_ind:    - vp6_ind + vp6_indc  = 0
 Act2_vp6_ind:    - vp6_indc + vp6_indd >= 0
 Inact1_vp6_ind:  vp6_ind + vp6_inda  = 1
 Inact2_vp6_ind:  - vp6_inda + vp6_indb >= 0
 Act1_vn6_ind:    vn6_indc - vn6_ind  = 0
 Act2_vn6_ind:    - vn6_indc + vn6_indd >= 0
 Inact1_vn6_ind:  vn6_ind + vn6_inda  = 1
 Inact2_vn6_ind:  - vn6_inda + vn6_indb >= 0
 Act1_vp7_ind:    vp7_indc - vp7_ind  = 0
 Act2_vp7_ind:    - vp7_indc + vp7_indd >= 0
 Inact1_vp7_ind:  vp7_ind + vp7_inda  = 1
 Inact2_vp7_ind:  - vp7_inda + vp7_indb >= 0
 Act1_vn7_ind:    vn7_indc - vn7_ind  = 0
 Act2_vn7_ind:    - vn7_indc + vn7_indd >= 0
 Inact1_vn7_ind:  vn7_ind + vn7_inda  = 1
 Inact2_vn7_ind:  - vn7_inda + vn7_indb >= 0
 Act1_vp8_ind:    - vp8_ind + vp8_indc  = 0
 Act2_vp8_ind:    - vp8_indc + vp8_indd >= 0
 Inact1_vp8_ind:  vp8_ind + vp8_inda  = 1
 Inact2_vp8_ind:  - vp8_inda + vp8_indb >= 0
 Act1_vn8_ind:    - vn8_ind + vn8_indc  = 0
 Act2_vn8_ind:    - vn8_indc + vn8_indd >= 0
 Inact1_vn8_ind:  vn8_ind + vn8_inda  = 1
 Inact2_vn8_ind:  - vn8_inda + vn8_indb >= 0
 vp0_ind_vn0_ind: vp0_ind + vn0_ind <= 1
 vp1_ind_vn1_ind: vp1_ind + vn1_ind <= 1
 vp2_ind_vn2_ind: vp2_ind + vn2_ind <= 1
 vp3_ind_vn3_ind: vp3_ind + vn3_ind <= 1
 vp4_ind_vn4_ind: vp4_ind + vn4_ind <= 1
 vp5_ind_vn5_ind: vp5_ind + vn5_ind <= 1
 vp6_ind_vn6_ind: vp6_ind + vn6_ind <= 1
 vp7_ind_vn7_ind: vp7_ind + vn7_ind <= 1
 vp8_ind_vn8_ind: vp8_ind + vn8_ind <= 1
 IntCut0:         vp0_ind + vn0_ind <= 0
 IntCut1:         vp0_ind + vn0_ind <= 0
 IntCut2:         vp8_ind + vn8_ind <= 0
 IntCut3:         vp4_ind + vn4_ind + vp7_ind + vn7_ind <= 1
 IntCut4:         vp4_ind + vn4_ind + vp7_ind + vn7_ind <= 1
 IntCut5:         vp4_ind + vn4_ind + vp6_ind + vn6_ind <= 1
 IntCut6:         vp4_ind + vn4_ind + vp5_ind + vn5_ind <= 1
 SolSz:           vp0_ind + vn0_ind + vp1_ind + vn1_ind + vp2_ind + vn2_ind
                  + vp3_ind + vn3_ind + vp4_ind + vn4_ind + vp5_ind + vn5_ind
                  + vp6_ind + vn6_ind + vp7_ind + vn7_ind + vp8_ind + vn8_ind
                   = 3
 IntCut7:         vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp7_ind + vn7_ind
                  <= 2
 IntCut8:         vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp7_ind + vn7_ind
                  <= 2
 IntCut9:         vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp7_ind + vn7_ind
                  <= 2
 IntCut10:        vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp6_ind + vn6_ind
                  <= 2
 IntCut11:        vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp6_ind + vn6_ind
                  <= 2
 IntCut12:        vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp5_ind + vn5_ind
                  <= 2
 IntCut13:        vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp5_ind + vn5_ind
                  <= 2
 Act3_vp0_ind:   vp0_indd = 1 -> vp0 - C >= 0
 Inact3_vp0_ind: vp0_indb = 1 -> vp0  = 0
 Act3_vn0_ind:   vn0_indd = 1 -> vn0 - C >= 0
 Inact3_vn0_ind: vn0_indb = 1 -> vn0  = 0
 Act3_vp1_ind:   vp1_indd = 1 -> vp1 - C >= 0
 Inact3_vp1_ind: vp1_indb = 1 -> vp1  = 0
 Act3_vn1_ind:   vn1_indd = 1 -> - C + vn1 >= 0
 Inact3_vn1_ind: vn1_indb = 1 -> vn1  = 0
 Act3_vp2_ind:   vp2_indd = 1 -> vp2 - C >= 0
 Inact3_vp2_ind: vp2_indb = 1 -> vp2  = 0
 Act3_vn2_ind:   vn2_indd = 1 -> vn2 - C >= 0
 Inact3_vn2_ind: vn2_indb = 1 -> vn2  = 0
 Act3_vp3_ind:   vp3_indd = 1 -> - C + vp3 >= 0
 Inact3_vp3_ind: vp3_indb = 1 -> vp3  = 0
 Act3_vn3_ind:   vn3_indd = 1 -> - C + vn3 >= 0
 Inact3_vn3_ind: vn3_indb = 1 -> vn3  = 0
 Act3_vp4_ind:   vp4_indd = 1 -> vp4 - C >= 0
 Inact3_vp4_ind: vp4_indb = 1 -> vp4  = 0
 Act3_vn4_ind:   vn4_indd = 1 -> vn4 - C >= 0
 Inact3_vn4_ind: vn4_indb = 1 -> vn4  = 0
 Act3_vp5_ind:   vp5_indd = 1 -> - C + vp5 >= 0
 Inact3_vp5_ind: vp5_indb = 1 -> vp5  = 0
 Act3_vn5_ind:   vn5_indd = 1 -> vn5 - C >= 0
 Inact3_vn5_ind: vn5_indb = 1 -> vn5  = 0
 Act3_vp6_ind:   vp6_indd = 1 -> vp6 - C >= 0
 Inact3_vp6_ind: vp6_indb = 1 -> vp6  = 0
 Act3_vn6_ind:   vn6_indd = 1 -> vn6 - C >= 0
 Inact3_vn6_ind: vn6_indb = 1 -> vn6  = 0
 Act3_vp7_ind:   vp7_indd = 1 -> vp7 - C >= 0
 Inact3_vp7_ind: vp7_indb = 1 -> vp7  = 0
 Act3_vn7_ind:   vn7_indd = 1 -> vn7 - C >= 0
 Inact3_vn7_ind: vn7_indb = 1 -> vn7  = 0
 Act3_vp8_ind:   vp8_indd = 1 -> vp8 - C >= 0
 Inact3_vp8_ind: vp8_indb = 1 -> vp8  = 0
 Act3_vn8_ind:   vn8_indd = 1 -> - C + vn8 >= 0
 Inact3_vn8_ind: vn8_indb = 1 -> vn8  = 0
Bounds
      u0 Free
      u1 Free
      u2 Free
      u5 Free
      u3 Free
      u4 Free
      C >= 1
 0 <= vp0_indc <= 1
 0 <= vp0_ind <= 1
 0 <= vp0_indd <= 1
 0 <= vp0_inda <= 1
 0 <= vp0_indb <= 1
 0 <= vn0_indc <= 1
 0 <= vn0_ind <= 1
 0 <= vn0_indd <= 1
 0 <= vn0_inda <= 1
 0 <= vn0_indb <= 1
 0 <= vp1_ind <= 1
 0 <= vp1_indc <= 1
 0 <= vp1_indd <= 1
 0 <= vp1_inda <= 1
 0 <= vp1_indb <= 1
 0 <= vn1_ind <= 1
 0 <= vn1_indc <= 1
 0 <= vn1_indd <= 1
 0 <= vn1_inda <= 1
 0 <= vn1_indb <= 1
 0 <= vp2_ind <= 1
 0 <= vp2_indc <= 1
 0 <= vp2_indd <= 1
 0 <= vp2_inda <= 1
 0 <= vp2_indb <= 1
 0 <= vn2_ind <= 1
 0 <= vn2_indc <= 1
 0 <= vn2_indd <= 1
 0 <= vn2_inda <= 1
 0 <= vn2_indb <= 1
 0 <= vp3_indc <= 1
 0 <= vp3_ind <= 1
 0 <= vp3_indd <= 1
 0 <= vp3_inda <= 1
 0 <= vp3_indb <= 1
 0 <= vn3_indc <= 1
 0 <= vn3_ind <= 1
 0 <= vn3_indd <= 1
 0 <= vn3_inda <= 1
 0 <= vn3_indb <= 1
 0 <= vp4_ind <= 1
 0 <= vp4_indc <= 1
 0 <= vp4_indd <= 1
 0 <= vp4_inda <= 1
 0 <= vp4_indb <= 1
 0 <= vn4_ind <= 1
 0 <= vn4_indc <= 1
 0 <= vn4_indd <= 1
 0 <= vn4_inda <= 1
 0 <= vn4_indb <= 1
 0 <= vp5_ind <= 1
 0 <= vp5_indc <= 1
 0 <= vp5_indd <= 1
 0 <= vp5_inda <= 1
 0 <= vp5_indb <= 1
 0 <= vn5_ind <= 1
 0 <= vn5_indc <= 1
 0 <= vn5_indd <= 1
 0 <= vn5_inda <= 1
 0 <= vn5_indb <= 1
 0 <= vp6_ind <= 1
 0 <= vp6_indc <= 1
 0 <= vp6_indd <= 1
 0 <= vp6_inda <= 1
 0 <= vp6_indb <= 1
 0 <= vn6_indc <= 1
 0 <= vn6_ind <= 1
 0 <= vn6_indd <= 1
 0 <= vn6_inda <= 1
 0 <= vn6_indb <= 1
 0 <= vp7_indc <= 1
 0 <= vp7_ind <= 1
 0 <= vp7_indd <= 1
 0 <= vp7_inda <= 1
 0 <= vp7_indb <= 1
 0 <= vn7_indc <= 1
 0 <= vn7_ind <= 1
 0 <= vn7_indd <= 1
 0 <= vn7_inda <= 1
 0 <= vn7_indb <= 1
 0 <= vp8_ind <= 1
 0 <= vp8_indc <= 1
 0 <= vp8_indd <= 1
 0 <= vp8_inda <= 1
 0 <= vp8_indb <= 1
 0 <= vn8_ind <= 1
 0 <= vn8_indc <= 1
 0 <= vn8_indd <= 1
 0 <= vn8_inda <= 1
 0 <= vn8_indb <= 1
Binaries
 vp0_indc  vp0_ind  vp0_indd  vp0_inda  vp0_indb  vn0_indc  vn0_ind  vn0_indd 
 vn0_inda  vn0_indb  vp1_ind  vp1_indc  vp1_indd  vp1_inda  vp1_indb  vn1_ind 
 vn1_indc  vn1_indd  vn1_inda  vn1_indb  vp2_ind  vp2_indc  vp2_indd  vp2_inda 
 vp2_indb  vn2_ind  vn2_indc  vn2_indd  vn2_inda  vn2_indb  vp3_indc  vp3_ind 
 vp3_indd  vp3_inda  vp3_indb  vn3_indc  vn3_ind  vn3_indd  vn3_inda  vn3_indb 
 vp4_ind  vp4_indc  vp4_indd  vp4_inda  vp4_indb  vn4_ind  vn4_indc  vn4_indd 
 vn4_inda  vn4_indb  vp5_ind  vp5_indc  vp5_indd  vp5_inda  vp5_indb  vn5_ind 
 vn5_indc  vn5_indd  vn5_inda  vn5_indb  vp6_ind  vp6_indc  vp6_indd  vp6_inda 
 vp6_indb  vn6_indc  vn6_ind  vn6_indd  vn6_inda  vn6_indb  vp7_indc  vp7_ind 
 vp7_indd  vp7_inda  vp7_indb  vn7_indc  vn7_ind  vn7_indd  vn7_inda  vn7_indb 
 vp8_ind  vp8_indc  vp8_indd  vp8_inda  vp8_indb  vn8_ind  vn8_indc  vn8_indd 
 vn8_inda  vn8_indb 
End
