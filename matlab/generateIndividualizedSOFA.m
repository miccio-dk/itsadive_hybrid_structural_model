function generateIndividualizedSOFA(sofa_path, hrirs, pos, subj, head_width_m)
    % create sofa file
    SOFAstart;
    Obj = SOFAgetConventions('SimpleFreeFieldHRIR');
    Obj.GLOBAL_AuthorContact = 'rmicci18@student.aau.dk';
    Obj.GLOBAL_Comment = 'Individualized HRTF based on synthesized pinna responses, shoulder effect from VIKING, and ITD matched from HUTUBS';
    Obj.GLOBAL_DatabaseName = 'Individualized HRTF';
    Obj.GLOBAL_EmitterDescription = '-';
    Obj.GLOBAL_History = '-';
    Obj.GLOBAL_License = 'Creative Commons Attribution 4.0 International license';
    Obj.GLOBAL_ListenerShortName = subj;
    Obj.GLOBAL_Organization = 'Aalborg University';
    Obj.GLOBAL_Origin = 'https://itsadive.create.aau.dk/';
    Obj.GLOBAL_ReceiverDescription = '-';
    Obj.GLOBAL_References = '-';
    Obj.GLOBAL_RoomDescription = '-';
    Obj.GLOBAL_RoomLocation = '-';
    Obj.GLOBAL_RoomType = 'free field';
    Obj.GLOBAL_SourceDescription = '-';
    Obj.GLOBAL_Title = 'HRTF';
    Obj.GLOBAL_ApplicationName = '-';
    Obj.GLOBAL_ApplicationVersion = SOFAgetVersion('API');
    Obj.ReceiverPosition = [0 head_width_m/2 0; 0 -head_width_m/2 0];
    Obj.ReceiverPosition_Units = 'metre';
    Obj.ReceiverPosition_Type = 'cartesian';
    Obj.Data.IR = hrirs; 
    Obj.SourcePosition = pos; 
    Obj = SOFAupdateDimensions(Obj);
    SOFAsave(sofa_path, Obj, 1); 
end