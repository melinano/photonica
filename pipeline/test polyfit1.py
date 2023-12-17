from typing import List, Any

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from sympy import symbols, Eq, solve, diff
import sympy as sp

# from gplearn.genetic import SymbolicRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
# import mpmath
#
# # 1. Parse the data
data = '''694	1079
695	1079
696	1080
697	1080
698	1081
699	1082
700	1082
701	1083
702	1083
703	1084
704	1084
705	1085
706	1085
707	1085
708	1086
709	1086
710	1086
711	1087
712	1087
713	1087
714	1087
715	1087
716	1088
717	1088
718	1088
719	1088
720	1088
721	1088
722	1088
723	1088
724	1088
725	1088
726	1088
727	1088
728	1088
729	1088
730	1088
731	1088
732	1088
733	1088
734	1087
735	1087
736	1087
737	1086
738	1086
739	1086
740	1085
741	1085
742	1085
743	1085
744	1084
745	1084
746	1084
747	1084
748	1083
749	1083
750	1083
751	1082
752	1082
753	1081
754	1081
755	1080
756	1080
757	1079
758	1079
759	1078
760	1078
761	1077
762	1077
763	1076
764	1075
765	1075
766	1074
767	1073
768	1072
769	1070
770	1067
771	1063
772	1060
773	1059
774	1057
775	1055
776	1054
777	1052
778	1051
779	1049
780	1048
781	1046
782	1045
783	1043
784	1041
785	1038
786	1034
787	1028
788	1021
789	1015
790	1012
791	1009
792	1007
793	1005
794	1002
795	1000
796	998
797	996
798	994
799	992
800	991
801	988
802	986
803	983
804	980
805	976
806	971
807	968
808	966
809	964
810	962
811	960
812	959
813	957
814	956
815	955
816	954
817	952
818	951
819	949
820	948
821	947
822	946
823	945
824	944
825	943
826	942
827	941
828	940
829	939
830	938
831	937
832	936
833	935
834	935
835	934
836	933
837	932
838	932
839	931
840	930
841	930
842	929
843	928
844	928
845	927
846	927
847	926
848	926
849	925
850	924
851	924
852	923
853	923
854	923
855	922
856	922
857	922
858	922
859	921
860	921
861	921
862	920
863	920
864	919
865	919
866	919
867	919
868	918
869	918
870	918
871	917
872	917
873	917
874	917
875	916
876	916
877	916
878	916
879	915
880	915
881	915
882	915
883	915
884	914
885	914
886	914
887	914
888	913
889	913
890	913
891	913
892	913
893	913
894	912
895	912
896	912
897	912
898	912
899	912
900	912
901	912
902	912
903	912
904	912
905	912
906	912
907	912
908	912
909	912
910	912
911	912
912	912
913	911
914	911
915	911
916	911
917	911
918	910
919	910
920	910
921	909
922	909
923	909
924	909
925	909
926	909
927	908
928	908
929	908
930	908
931	907
932	907
933	907
934	907
935	907
936	907
937	907
938	906
939	906
940	906
941	906
942	906
943	906
944	906
945	906
946	906
947	906
948	906
949	905
950	905
951	905
952	905
953	905
954	905
955	905
956	905
957	905
958	905
959	905
960	905
961	904
962	904
963	904
964	904
965	904
966	904
967	904
968	904
969	904
970	904
971	904
972	904
973	904
974	904
975	904
976	904
977	904
978	904
979	904
980	904
981	904
982	904
983	904
984	904
985	904
986	904
987	904
988	904
989	904
990	904
991	904
992	904
993	904
994	904
995	904
996	904
997	904
998	904
999	904
1000	904
1001	904
1002	904
1003	904
1004	904
1005	904
1006	904
1007	904
1008	904
1009	904
1010	904
1011	904
1012	904
1013	904
1014	904
1015	904
1016	904
1017	904
1018	904
1019	904
1020	904
1021	904
1022	904
1023	904
1024	904
1025	904
1026	904
1027	904
1028	904
1029	904
1030	904
1031	904
1032	904
1033	905
1034	905
1035	905
1036	905
1037	905
1038	905
1039	905
1040	906
1041	906
1042	906
1043	906
1044	907
1045	907
1046	907
1047	908
1048	908
1049	909
1050	909
1051	909
1052	910
1053	910
1054	910
1055	911
1056	911
1057	911
1058	912
1059	912
1060	913
1061	913
1062	913
1063	914
1064	914
1065	915
1066	915
1067	916
1068	916
1069	916
1070	917
1071	917
1072	918
1073	918
1074	919
1075	919
1076	920
1077	920
1078	920
1079	921
1080	922
1081	922
1082	923
1083	923
1084	924
1085	924
1086	925
1087	925
1088	926
1089	927
1090	927
1091	928
1092	929
1093	930
1094	931
1095	931
1096	932
1097	933
1098	934
1099	935
1100	935
1101	936
1102	937
1103	938
1104	939
1105	940
1106	941
1107	942
1108	943
1109	944
1110	945
1111	946
1112	946
1113	947
1114	948
1115	950
1116	951
1117	952
1118	952
1119	953
1120	954
1121	955
1122	956
1123	957
1124	958
1125	959
1126	960
1127	961
1128	962
1129	964
1130	965
1131	966
1132	967
1133	968
1134	969
1135	970
1136	971
1137	972
1138	973
1139	974
1140	975
1141	976
1142	977
1143	978
1144	979
1145	979
1146	980
1147	981
1148	982
1149	982
1150	983
1151	984
1152	985
1153	985
1154	986
1155	987
1156	987
1157	988
1158	989
1159	989
1160	990
1161	990
1162	991
1163	991
1164	992
1165	992
1166	993
1167	994
1168	994
1169	995
1170	995
1171	996
1172	996
1173	997
1174	998
1175	999
1176	999
1177	1000
1178	1001
1179	1003
1180	1009
1181	1016
1182	1023
1183	1025
1184	1027
1185	1028
1186	1029
1187	1030
1188	1032
1189	1033
1190	1035
1191	1036
1192	1037
1193	1038
1194	1039
1195	1041
1196	1043
1197	1046
1198	1053
1199	1060
1200	1066
1201	1067
1202	1069
1203	1070
1204	1071
1205	1072
1206	1073
1207	1073
1208	1074
1209	1075
1210	1076
1211	1077
1212	1078
1213	1079
1214	1080
1215	1080
1216	1081
1217	1081
1218	1082
1219	1082
1220	1083
1221	1083
1222	1084
1223	1084
1224	1084
1225	1084
1226	1084
1227	1085
1228	1085
1229	1085
1230	1085
1231	1085
1232	1086
1233	1086
1234	1086
1235	1086
1236	1086
1237	1086
1238	1086
1239	1086
1240	1086
1241	1086
1242	1086
1243	1086
1244	1086
1245	1086
1246	1086
1247	1086
1248	1086
1249	1086
1250	1086
1251	1086
1252	1086
1253	1086
1254	1086
1255	1086
1256	1086
1257	1086
1258	1086
1259	1086
1260	1086
1261	1086
1262	1086
1263	1085
1264	1085
1265	1084
1266	1084
1267	1083
1268	1082
1269	1082
1270	1081
1271	1080
1272	1080
1273	1079
1274	1078
1275	1077
1276	1076
1277	1076
1278	1075
1279	1074
1280	1073
1281	1072
1282	1071
1283	1071
1284	1070'''

# def normalize(data):
#     """
#     Normalize data to the range [0, 1].
#     Returns normalized data and the original data's min and max.
#     """
#     data_min = min(data)
#     data_max = max(data)
#     normalized_data = [(x - data_min) / (data_max - data_min) for x in data]
#     return normalized_data, data_min, data_max
#
# def denormalize(normalized_data, data_min, data_max):
#     """
#     Convert normalized data back to its original scale.
#     """
#     return [x * (data_max - data_min) + data_min for x in normalized_data]



lines = data.split("\n")
x = [int(line.split("\t")[0]) for line in lines]
# x , x_min, x_max = normalize(x)
print(x)
y = [float(line.split("\t")[1]) for line in lines]

# y, y_min, y_max = normalize(y)
y = [el - 1 for el in y]
print(y)
# plt.figure(figsize=(10, 6))
#plt.plot(x, y, 'b.', label='Original Data')
#plt.show()

# 2. Use polyfit for symbolic regression
coefficients = np.polyfit(x, y,
                          deg=5,
                          rcond=None,
                          full=False,
                          w=None,
                          cov=False)
y_pred = np.polyval(coefficients, x)

# 3. Plot the results
y = np.array(y)
x = np.array(x)
y_pred = np.array(y_pred)
coefficients = np.array(coefficients)
#print('y_pred: ', y_pred)

print('Значения коэффициентов: ', coefficients)
#equation = f"Функция аппроксимации: {coefficients[0]}x**5 + {coefficients[1]}x**4 + {coefficients[2]}x**3 + {coefficients[3]}x**2 + {coefficients[4]}x + {coefficients[5]}"
equation = f"Функция аппроксимации: {('-' if coefficients[0] < 0 else '+')} " \
           f"{abs(coefficients[0])}x**5 {('-' if coefficients[1] < 0 else '+')} " \
           f"{abs(coefficients[1])}x**4 {('-' if coefficients[2] < 0 else '+')} " \
           f"{abs(coefficients[2])}x**3 {('-' if coefficients[3] < 0 else '+')} " \
           f"{abs(coefficients[3])}x**2 {('-' if coefficients[4] < 0 else '+')} " \
           f"{abs(coefficients[4])}x {('-' if coefficients[5] < 0 else '+')} " \
           f"{abs(coefficients[5])}"
print(equation)


function = np.poly1d(coefficients)
print("Функция аппроксимации: ", "\n", function)


def relative_error(y_pred, y):
    diff = np.abs(y - y_pred)
    return diff / np.abs(y) * 100


a = relative_error(y, y_pred)
b = np.max(a)

print('Относительная ошибка аппроксимации:', b)

# plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b.', label='Original Data')
# plt.plot(x, y_pred, 'r-', label='Symbolic Regression')
#
# plt.legend()
plt.show()


max_y = max(y)
min_y = min(y)
max_diff = max_y - min_y
print('Глубина структуры:', max_diff)




# data_c = list(zip(x, y))
# print(data_c)




def remove_adjacent_duplicates(x, y):
    x_result = []
    y_result = []

    for i in range(len(x)):
        if i == 0 or x[i] != x[i - 1] and y[i] != y[i - 1]:
            x_result.append(x[i])
            y_result.append(y[i])

    return x_result, y_result

new_x, new_y = remove_adjacent_duplicates(x, y)
#print(new_y)
#plt.plot(new_x, new_y, 'b.', label='Original Data')
#plt.show()

dy_dx = np.gradient(new_y, new_x)
derivative = np.diff(dy_dx)
# print(dy_dx)

indices = np.where(np.diff(np.sign(dy_dx)))[0]

x_ind2 = new_x[indices[2]]
x_ind1 = new_x[indices[1]]
x_ind0 = new_x[indices[0]]
distance = x_ind2 - x_ind0
#     print(f"Расстояние между верхними точками перегиба: {distance}")

#print('indices:',indices)
#print('Диаметр модификаций:', x_ind0, x_ind1, x_ind2, '=', distance)
print('Диаметр модификаций:', distance)


# plt.plot(x, y, 'b.', label='Original Data')
# plt.plot(new_x, new_y, 'b.', label='New Data')
# plt.plot(fft_result.real, fft_result.imag)
# plt.show()






# string_y = y.astype(str)
# filename = 'data1.csv'
# with open(filename, 'w', newline='') as file:
#     writer = csv.writer(file)
#     for row in string_y:
#         writer.writerow(row)
# print(f'Данные успешно записаны в файл {filename}')
