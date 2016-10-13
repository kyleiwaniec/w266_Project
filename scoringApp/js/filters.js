iq4scoring
.filter('role', function () {
    return function (value) {
        switch(value){

            case 51 : return "Subject Matter Expert";
            break;
            case 52 : return "Faculty";
            break;
            case 53 : return "Student";
            break;
            case 54 : return "Project Lead";
            break;
            default: return "Staff"
        }
      
    };
})