import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update(mpl.rcParamsDefault)

r0=[0.9231597238080611,0.9254921566220708,0.9347435737790707,0.9283224663584301,0.9257150432559942,0.9429949464018978,0.9252821587010638,0.9296394159155638,0.9334673162814343,0.9320965929288485]
r1=[0.9187836305618204,0.9215829146057074,0.9327717444399993,0.9230373917888458,0.9206870429886851,0.9395994222274314,0.9196480231850563,0.9255034869814929,0.9279359714186085,0.9266602765471313]
r2=[0.9122594268653856,0.9154962344756885,0.924929892437393,0.9184607430318994,0.9151806937764253,0.9302638977845964,0.9122157728162741,0.9212376521567005,0.9212275892816723,0.9206951942519103]
r3=[0.9085135251589959,0.9102758260829962,0.9212475179434304,0.9097581680614618,0.9118382992654246,0.9252065779773023,0.9098219419672804,0.9152724278954953,0.9150576301108483,0.9165305651094097]
r4=[0.9025335444967123,0.9011646386160863,0.9269184038629705,0.9020432141554604,0.9077793657022892,0.9197873944602805,0.9009750683008146,0.9046989682800345,0.9080947764724596,0.9222069915134422]
r5=[0.9077852576871218,0.9081429137207487,0.9251704890436926,0.9072693325772807,0.9105250956600285,0.9187398637391359,0.9049702424587536,0.9057641335811051,0.9130219042741633,0.9197914441501778]
r6=[0.9068741009206487,0.9082792067123946,0.9212914423012375,0.9085372280507564,0.9097770820370421,0.921189662890987,0.9061631107246749,0.9091326848345618,0.9165476220578146,0.9123925031697264]
r7=[0.9048663264720755,0.9091646131355152,0.9233316172934225,0.9044782516707832,0.9071655480522102,0.9287429753047953,0.9102474744261412,0.9160565983874599,0.9110043241376079,0.9211467367970803]
r8=[0.9074209047695002,0.9079838793735154,0.922608200632529,0.9007433572476232,0.9068926068521543,0.9282354473570852,0.90829735016045,0.9138367759312804,0.9123649805627104,0.9145093467588952]
r9=[0.90944576557874,0.9085063162617926,0.9264817343122551,0.8992816296537248,0.9043411832591562,0.9260255702877485,0.9102026909042117,0.9149585715169398,0.9031636607400544,0.9147594021411979]
r10=[0.9057795029346314,0.907711184238912,0.9232439606152648,0.9110451629489843,0.9098904569620305,0.9300876864087134,0.900680938843109,0.9169967178073164,0.9142212693191698,0.9187736626867575]
r11=[0.9086955008293555,0.905389959470149,0.9212475179434304,0.9095547918741057,0.9100944962353639,0.929977537343747,0.9106504270181139,0.9159670128805821,0.9166152914161008,0.9109328931713961]
r12=[0.9089683960257061,0.9055722313446556,0.9214671187946684,0.9110677255791734,0.9100491581280681,0.930946401830665,0.910695188522578,0.9143754095706212,0.9128633797648907,0.913622235293316]
r13=[0.9001947173253846,0.9081429137207487,0.9230686222867223,0.9048872402748505,0.9133529255414621,0.9179143709132821,0.9156723728179658,0.9152500131520119,0.913881986082235,0.9174375274461292]
r14=[0.8980569486779684,0.9017139060209743,0.9219061635683864,0.8958463598980988,0.911951417937381,0.9271532657373933,0.9042267182142409,0.9104630961224908,0.9123423195843389,0.914168251917759]
r15=[0.9033575671791841,0.8992395606420278,0.9049833259956804,0.8929505201717989,0.9072337705218592,0.9299995682004238,0.9109637113615323,0.910418029269543,0.9186655598476242,0.8995637295524319]
r16=[0.8942977795558471,0.8977470839295868,0.9164471420630502,0.8996701326820796,0.9094822410888058,0.9275951250131736,0.9092841426343621,0.9162357431233834,0.908959520472588,0.9064711542164374]
r17=[0.9025564441743436,0.8937395728971629,0.9105777446604417,0.8958234138955106,0.9020570283101927,0.9305501701030631,0.9089927018089603,0.9007218506783948,0.9177649071214438,0.9038525512850005]
r18=[0.8962139280115573,0.9017596632029499,0.9063461020589123,0.8942157305341634,0.8972183821225929,0.9219011093747078,0.9091272245685171,0.9176452857761213,0.9123649805627104,0.902056495730923]
r19=[0.9002865505026014,0.8968274022007661,0.9058325092426891,0.8984583628076833,0.8984591897591402,0.9223232710703846,0.9058256658783551,0.907099485419534,0.9040559908187268,0.8987775389592196]
r20=[0.8962600497174783,0.9002484482220224,0.9050280397152164,0.898984425817374,0.8963902228323172,0.9207669814483755,0.902195810390944,0.9095387793116474,0.9078215282887994,0.9004646911530216]
r21=[0.8989080677609442,0.8957686009050481,0.9056091171681577,0.8945604773427821,0.892214773756109,0.9137981422320592,0.8938208444315465,0.905900021982377,0.9000220528779039,0.9103164182692698]
r22=[0.8995056696516425,0.8948238307439839,0.908664810822272,0.9016101367430653,0.8992395606420278,0.9146721507836082,0.8960982147261257,0.9063754709488583,0.9088912811220979,0.903622487252133]
r23=[0.8964214570007952,0.898344372160205,0.9070378757230125,0.8978862056059211,0.8945701860623988,0.918985137530954,0.8949830273657633,0.8994454589334084,0.9043303767800709,0.8994019229624814]
r24=[0.897619573912523,0.9022171073876003,0.9061005070323525,0.8944915386105932,0.8925615477494412,0.9168200067441983,0.8982336780708431,0.8969100996753917,0.9006191129638501,0.9027246772857742]
r25=[0.9001487972234685,0.8962291029922753,0.9094215679299954,0.8954791532997342,0.8938088201426778,0.9150080856225427,0.8982790583057002,0.8976646031237372,0.9010322304910767,0.9021717347538046]
r26=[0.8941128735723642,0.8949391000163506,0.9069486442544521,0.8987557406044916,0.8952617750223346,0.9117105867908667,0.8958252359785063,0.898418472930556,0.9028431220007689,0.9014570153362843]
r27=[0.8964445128117325,0.8984132644804147,0.905631458855296,0.8982524281991469,0.8943164694424965,0.9148513313819838,0.8964393212647489,0.9028603088182884,0.8997233742556311,0.8964612175497936]
r28=[0.8982870603861234,0.895699505171765,0.905631458855296,0.8973136835776742,0.8933008823535324,0.911081138633425,0.8958252359785063,0.9015869445294071,0.9029576131842567,0.9002568569671107]
r29=[0.8951986492913859,0.8985510332748167,0.9061228366036074,0.8952266113909241,0.8948468857863257,0.9130355034277621,0.8962346729205757,0.9004712736611797,0.9051987169502483,0.9001182741830636]
r30=[0.8956141296965838,0.8972873608945607,0.9059665180472158,0.9031818992690466,0.8959298035596428,0.9115532654997757,0.894345153111582,0.9026557821344987,0.9008715961862069,0.9039445605030619]


plt.figure(1)
data = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
        r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
        r21, r22, r23, r24, r25, r26, r27, r28, r29, r30]
plt.title(r"Boxplot of RMSE for NMF for all splits")
plt.xlabel(r"Level of truncation $r$")
plt.ylabel("RMSE")
plt.boxplot(data)
plt.show()
