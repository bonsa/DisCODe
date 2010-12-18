install(FILES CameraViewer.xml MovieViewer.xml MovieWriter.xml MS_Barcode.xml MS_Sign.xml SequenceViewer.xml TriggeredSequenceViewer.xml
  DESTINATION share/Tasks
  COMPONENT tasks)

FILE(GLOB KW_Tasks KW/*.xml)  
install(FILES ${KW_Tasks}
  DESTINATION share/Tasks/KW
  COMPONENT tasks)