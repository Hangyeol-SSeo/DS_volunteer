# Data Science 2023-1 Term project
### Team member: 서한결, 이아영, 이은섭, 박현서

# Topic
Among teenagers who want to start volunteer work, there are not many cases where they are experienced or familiar with volunteer work. In addition, there are often cases where volunteer activities are canceled in the middle or the start of activities is failed at all. Therefore, we would like to conduct an analysis aimed at these teenagers that can help them choose volunteer activities successfully with responsibility. For example, compare cancelled/deleted activities with active/completed activities to predict the outcome of the activity and calculate the cancellation probability.

## Dataset
**Volunteer work dataset**</br>
https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=4174bb76-1077-4e52-a84b-e341397c7c74

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


### Step4. Data preprocessing

### Step5. Data analysis

### Step6. Data evaluation
