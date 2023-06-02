# Data Science 2023-1 Term project
### Team member: 서한결, 이아영, 이은섭, 박현서

# Topic
Among teenagers who want to start volunteer work, there are not many cases where they are experienced or familiar with volunteer work. In addition, there are often cases where volunteer activities are canceled in the middle or the start of activities is failed at all. Therefore, we would like to conduct an analysis aimed at these teenagers that can help them choose volunteer activities successfully with responsibility. For example, compare cancelled/deleted activities with active/completed activities to predict the outcome of the activity and calculate the cancellation probability.

## Dataset
**Volunteer work dataset**</br>
https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=4174bb76-1077-4e52-a84b-e341397c7c74

## Samples

## Columns used
0. : 프로그램일련번호 - PROGRM_SEQ_NO / DECIMAL / 12,0
1. : 등록구분코드명 - REGIST_SDIV_CD_NM / VARCHAR / 300
2. : 상세유형코드명 - DETAIL_TY_CD_NM / VARCHAR / 300
3. : 활동시도코드명 - ACT_CTPRVN_CD_NM / VARCHAR / 300
4. : 활동영역코드 - ACT_RELM_CD / VARCHAR / 300
5. : 상세내용코드명 - DETAIL_CN_CD_NM / VARCHAR / 300
6. : 활동시군구코드명 - ACT_SIGNGU_CD_NM / VARCHAR / 300
7. : 인증시간내용 - CRTFC_TIME_CN / DECIMAL / 5,0
8. : 참가비용 - PARTCPT_CT / DECIMAL / 9,0
9. : 회차 - TME / DECIMAL / 5,0
10. : 상태코드명 - STATE_CD_NM / VARCHAR / 300
11. : 모집인원구분코드명 - RCRIT_NMPR_SDIV_CD_NM / VARCHAR / 300
12. : 모집인원수 - RCRIT_NMPR_CO / DECIMAL / 10,0
13. : 활동시작시간 - ACT_BEGIN_TIME / VARCHAR / 10
14. : 활동종료시간 - ACT_END_TIME / VARCHAR / 10

## Steps of End-to-end Big data process
### Step1. Objective setting
What is the _difference_ between activities canceled/deleted compared to activities that are active/completed? </br>
For example, compare the characteristics of activity type, region, time, etc </br>
=> This data allows those who want to start volunteering to determine whether they can successfully complete their volunteer work. </br>
Target: Among teenagers who want to start volunteer work, there are not many cases where they are experienced or familiar with volunteer work. In addition, there are often cases where volunteer activities are canceled in the middle or failed to start activities at all. Therefore, it analyzes which volunteer activities are mainly canceled and completed for these teenagers to help them choose volunteer activities successfully with responsibility.

### Step2. Data curation(skip)
https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=4174bb76-1077-4e52-a84b-e341397c7c74

### Step3. Data inspection
#### Full data description
|index|PROGRM\_SEQ\_NO|REGIST\_SDIV\_CD|DETAIL\_TY\_CD|REQST\_POSBL\_BEGIN\_DE|REQST\_POSBL\_END\_DE|CANCL\_POSBL\_BEGIN\_DE|CANCL\_POSBL\_END\_DE|ACT\_CTPRVN\_CD|ONE\_TME\_PROGRM\_SEQ\_NO|INSTT\_SEQ\_NO|INSTT\_CLUB\_SEQ\_NO|ACT\_RELM\_CD\_NM|DETAIL\_CN\_CD|BSNS\_CD|REGIST\_CTPRVN\_CD|ACT\_SIGNGU\_CD|CRTFC\_TIME\_CN|PARTCPT\_CT|TME|GRP\_SDIV\_CD|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|211575\.0|211575\.0|209415\.0|211575\.0|211575\.0|211575\.0|211575\.0|211575\.0|211575\.0|211575\.0|211575\.0|211575\.0|209320\.0|5\.0|211575\.0|211564\.0|211575\.0|211575\.0|211575\.0|989\.0|
|mean|8791040\.1088881|110001\.94804206546|114001\.43363655898|20202688\.51162472|20203464\.4014652|20203056\.76832329|20203464\.730705425|20007\.82602859506|993083\.8030438379|49023\.27213990311|24939\.295805269998|113001\.00938674228|115001\.74495987006|117049\.0|20007\.82602859506|29478\.925715150024|3\.33719484816259|20\.18214817440624|1\.1614557485525228|119002\.12335692618|
|std|78735\.98408312947|0\.5502693065095743|0\.8011288560545596|96332\.11216619099|40267\.89225622339|39631\.57860634246|40268\.206561782776|5\.050174327086214|2786015\.1931557665|8310\.217516388126|45383\.508817950285|0\.09642963916209933|1\.3107243353596485|0\.0|5\.050174327086214|19397\.31035710573|1\.6094962719491144|3133\.8148499745876|1\.6056900622515347|0\.6558290909709796|
|min|904530\.0|110001\.0|114001\.0|30\.0|2140803\.0|2140803\.0|2140803\.0|20001\.0|0\.0|0\.0|0\.0|113001\.0|115001\.0|117049\.0|20001\.0|21001\.0|1\.0|0\.0|0\.0|119001\.0|
|25%|8738763\.5|110002\.0|114001\.0|20200305\.0|20200522\.0|20200305\.0|20200522\.0|20003\.0|0\.0|50220\.0|0\.0|113001\.0|115001\.0|117049\.0|20003\.0|23003\.0|2\.0|0\.0|1\.0|119002\.0|
|50%|8792419\.0|110002\.0|114001\.0|20200830\.0|20201003\.0|20200831\.0|20201003\.0|20008\.0|0\.0|51374\.0|0\.0|113001\.0|115001\.0|117049\.0|20008\.0|28012\.0|3\.0|0\.0|1\.0|119002\.0|
|75%|8845528\.5|110002\.0|114001\.0|20210105\.0|20210217\.0|20210105\.0|20210217\.0|20013\.0|0\.0|55073\.0|0\.0|113001\.0|115002\.0|117049\.0|20013\.0|33008\.0|4\.0|0\.0|1\.0|119003\.0|
|max|8898490\.0|110004\.0|114004\.0|20601226\.0|20610124\.0|20610118\.0|20610124\.0|20018\.0|8898458\.0|56161\.0|111825\.0|113002\.0|115005\.0|117049\.0|20018\.0|222025\.0|8\.0|999999\.0|16\.0|119003\.0|

#### Selected data example
|index|PROGRM\_SEQ\_NO|REGIST\_SDIV\_CD\_NM|DETAIL\_TY\_CD\_NM|ACT\_CTPRVN\_CD\_NM|ACT\_RELM\_CD|DETAIL\_CN\_CD\_NM|ACT\_SIGNGU\_CD\_NM|CRTFC\_TIME\_CN|PARTCPT\_CT|TME|STATE\_CD\_NM|RCRIT\_NMPR\_SDIV\_CD\_NM|RCRIT\_NMPR\_CO|ACT\_BEGIN\_DT|ACT\_BEGIN\_TIME|ACT\_END\_DT|ACT\_END\_TIME|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|6740176|일괄|지역사회봉사활동|서울특별시|봉사활동|교육\(지도\)봉사|동작구|8|0|0|활동완료|전체|6|20160618|13:30|20160618|21:30|
|1|6740034|일괄|지역사회봉사활동|서울특별시|봉사활동|교육\(지도\)봉사|동작구|8|0|0|삭제|개별|0|20160618|13:30|20160618|21:30|
|2|6740035|일괄|지역사회봉사활동|서울특별시|봉사활동|교육\(지도\)봉사|동작구|8|0|0|삭제|전체|1|20160618|13:30|20160618|21:30|
|3|6740037|일괄|지역사회봉사활동|서울특별시|봉사활동|교육\(지도\)봉사|동작구|8|0|0|삭제|전체|1|20160618|13:30|20160618|21:30|
|4|6013846|개별|지역사회봉사활동|서울특별시|봉사활동|재능봉사|NaN|4|0|0|활동취소|전체|3|2140803|13:00|2140803|17:00|
|5|920667|개별|지역사회봉사활동|부산광역시|봉사활동|교육\(지도\)봉사|NaN|4|0|0|삭제|전체|10|20610125|14:00|20610125|18:00|
|6|913640|개별|지역사회봉사활동|부산광역시|봉사활동|교육\(지도\)봉사|NaN|4|0|0|삭제|전체|10|20610125|14:00|20610125|18:00|
|7|912171|개별|지역사회봉사활동|부산광역시|봉사활동|교육\(지도\)봉사|NaN|4|0|0|삭제|전체|1|20610124|13:00|20610125|17:00|
|8|904530|개별|지역사회봉사활동|부산광역시|봉사활동|교육\(지도\)봉사|NaN|4|0|0|삭제|개별|4|20610122|1:00|20610122|5:00|
|9|7371820|개별|지역사회봉사활동|대구광역시|봉사활동|교육\(지도\)봉사|남구|1|0|0|활동취소|전체|1|20270520|17:00|20270520|18:00|

### Step4. Data preprocessing
#### Data selection: 
Among a total of 63 columns, features that meet our purpose were selected through meetings with team members. </br>
We decided to select and use 15 columns out of the total feature</br>



### Step5. Data analysis

### Step6. Data evaluation

## Others
### relevant information outside machine learning
