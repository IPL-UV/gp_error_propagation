# Accounting for Input Noise in Gaussian Process Parameter Retrieval

## [Juan Emmanuel Johnson][3], [Valero Laparra][2], [Gustau Camps-Valls][1]

### Abstract

Gaussian   Processes   (GPs)   are   a   class   of   kernelmethods   that   have   shown   to   be   very   useful   in   geoscienceand  remote  sensing  applications  for  parameter  retrieval,  modelinversion  and  emulation.  They  are  widely  used  because  theyare  simple,  flexible,  and  provide  accurate  estimates.  GPs  arebased  on  a  Bayesian  statistical  framework  which  provides  aposterior  probability  function  for  each  estimation.  Therefore,besides  the  usual  prediction  (given  in  this  case  by  the  meanfunction),  GPs  come  equipped  with  the  possibility  to  obtain  apredictive variance (i.e. error bars, confidence intervals) for eachprediction.  Unfortunately,  the  GP  formulation  usually  assumesthat  there  is  no  noise  in  the  inputs,  only  in  the  observations.However, this is often not the case in Earth observation problemswhere   an   accurate   assessment   of   the   measuring   instrumenterror  is  typically  available,  and  where  there  is  huge  interestin  characterizing  the  error  propagation  through  the  processingpipeline.  In  this  paper,  we  demonstrate  how  one  can  accountfor  input  noise  estimates  using  a  GP  model  formulation  whichpropagates the error terms using the derivative of the predictivemean function. We analyze the resulting predictive variance termand show how they more accurately represent the model error ina temperature prediction problem from infrared sounding data.


---
Full Text: Arxiv
Code: [github.com/IPL-UV/gp_error_propagation][4]


[1]: https://www.uv.es/gcamps/
[2]: https://www.uv.es/lapeva/
[3]: https://jejjohnson.github.io/academic/
[4]: https://github.com/IPL-UV/gp_error_propagation
