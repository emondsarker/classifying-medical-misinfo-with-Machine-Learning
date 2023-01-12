import './App.css';
import react, {useState, useEffect} from 'react'
import Grid from '@mui/material/Grid';
import axios from 'axios';
import { TextField, Autocomplete, Button } from '@mui/material';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert, { AlertProps } from '@mui/material/Alert';

import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CancelIcon from '@mui/icons-material/Cancel';
import Dialog from '@mui/material/Dialog';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';


function App() {

  const [name, setName] = useState("");
  const [prediction, setPrediction] = useState("");
  const [open, setOpen] = useState(false)
  const [open2, setOpen2] = useState(false)
  const [open3, setOpen3] = useState(false)
  const [open4, setOpen4] = useState(false)
  const [language, setLanguage] = useState('');
    
    useEffect(() => {
       
    }, [])
    
    async function submit() {
      console.log(name)
      if(language == 'English'){
        axios.post('http://localhost:8000/predict', {
        message: name,
         })
        .then(function (response) {
        console.log(response.data.prediction);
        if(response.data.prediction == '[0]'){
          setPrediction("Valid")
          setOpen2(true)
        }
        if(response.data.prediction == '[1]'){
          setPrediction("Invalid")
          setOpen(true)
        }
        })
        .catch(function (error) {
        console.log(error);
        });
      }

      if(language == 'Bangla'){
        axios.post('http://localhost:8000/predict2', {
        message: name,
         })
        .then(function (response) {
        console.log(response.data.prediction);
        if(response.data.prediction == '[0]'){
          setPrediction("Valid")
          setOpen4(true)
        }
        if(response.data.prediction == '[1]'){
          setPrediction("Invalid")
          setOpen3(true)
        }
        })
        .catch(function (error) {
        console.log(error);
        });
      }
      
    }


    const handleClose =()=>{
      setOpen(false);
    }
    const handleClose2 =()=>{
      setOpen2(false);
    }
    const handleClose3 =()=>{
      setOpen3(false);
    }
    const handleClose4 =()=>{
      setOpen4(false);
    }
   
    const handleChange = (event) => {
      setLanguage(event.target.value);
    };

    
  

  return (
    <>
    <Dialog onClose={handleClose} open={open} maxWidth='md'>
      <div className='title2'>
          <h1>
            <CancelIcon sx={{fontSize: 330}}/>
          </h1>
          <h1>Misinformation Detected</h1>
          <p>There is a high likelihood of your text being misinformation, please verify your sources.</p>
        </div>
      
    </Dialog>

    <Dialog onClose={handleClose2} open={open2} maxWidth='md'>
      <div className='title3'>
          <h1>
            <CheckCircleOutlineIcon sx={{fontSize: 330}}/>
          </h1>
          <h1>Seems Valid :)</h1>
          <p>Everything seems okay however, please verify your sources to be safe.</p>
        </div>
      
    </Dialog>

    <Dialog onClose={handleClose3} open={open3} maxWidth='md'>
      <div className='title2'>
          <h1>
            <CancelIcon sx={{fontSize: 330}}/>
          </h1>
          <h1>ভুল তথ্য সনাক্ত করা হয়েছে!</h1>
          <p>আপনার টেক্সট ভুল তথ্য হওয়ার সম্ভাবনা আছে, আপনার সূত্র যাচাই করুন।</p>
        </div>
      
    </Dialog>

    <Dialog onClose={handleClose4} open={open4} maxWidth='md'>
      <div className='title3'>
          <h1>
            <CheckCircleOutlineIcon sx={{fontSize: 330}}/>
          </h1>
          <h1>সঠিক তথ্য! :)</h1>
          <p>সবকিছু ঠিক আছে বলে মনে হচ্ছে, নিরাপদ হতে আপনার সূত্র যাচাই করুন।</p>
        </div>
      
    </Dialog>


    <div id="loader"></div>
<div>
    
    <div className='body'>
    <div className="">  
    <Grid container spacing={0}>

      <Grid item lg={12}>
        <div className='title'>
          <h1>
            <MedicalInformationIcon sx={{fontSize: 270}}/>
          </h1>
          <h1><br/><br/>Medical Misinformation Detector<br/>স্বাস্থ্য সম্পর্কিত ভুল তথ্য সনাক্তকারী</h1>
         
        </div>
      </Grid>
      <Grid item lg={12}>


      


      <h3 className='text'>Enter what text you want to verify here. | Example: Be aware, covid-19 vaccines may give your children autism</h3>
      <h3 className='text'>      আপনি এখানে কোন টেক্সট যাচাই করতে চান তা লিখুন। | উদাহরণ: সচেতন থাকুন, কোভিড-১৯ টিকা আপনার বাচ্চাদের অটিজম দিতে পারে।</h3>

        <div className='inputbox'>
        <Box sx={{ minWidth: 120 }}>
      <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">Language / ভাষা</InputLabel>
        <Select
          labelId="demo-simple-select-label"
          id="demo-simple-select"
          value={language}
          label="Age"
          onChange={handleChange}
        >
          <MenuItem value={'English'}>English</MenuItem>
          <MenuItem value={'Bangla'}>বাংলা</MenuItem>
        </Select>
      </FormControl>
    </Box>
          <br></br>
          <TextField id="standard-basic" 
          required='true'
          fullWidth
          label="Paste your content here / আপনার টেক্সট এখানে লিখুন।" 
          type="name"
          autoComplete='name'
          variant="standard" 
          value={name}
          onChange={(e) => setName(e.target.value)}
          />
        <Button
        fullWidth
        onClick={()=>{submit()}}>
          verify / যাচাই করুন
        </Button>
        </div>
      </Grid>
    </Grid>
    </div>
    </div>
    </div>
    <div className='info'>
      <div className='border'>
      <Grid container spacing={5}>
        <Grid item lg={4}>
            <div className='textbox'>
              <p>
              <b>
                What is Medical Misinformation?
              </b>
              </p>
              <p>
                Medical Misinformation is considered any form of misleading or harmful information regarding health or medicine.
              </p>
              <p>
                There can be a few different categories of misinformation found online, the most common ones are related to: vaccine, medicine, nutrition, sexual well-being.
              </p>
            </div>
        </Grid>  
      
      
        <Grid item lg={4}>
            <div className='textbox'>
              <p>
              <b>
                How does the detector work?
              </b>
              </p>
              <p>
                The Medical Misinformation Detector is built upon power Deep Learning and Machine Learning models that have been trained to classify a wide variety of medical misinformation.
              </p>
              <p>
                Enter any content text content you found online into the text box above and you will get an analysis of whether or not the content is misnformation or not.
              </p>
            </div>
        </Grid>
        <Grid item lg={4}>
            <div className='textbox'>
            <p>
              <b>
                About Us
              </b>
              </p>
              <p>
                We are a group of undergrad students from North South University that are eager to bring a postiive impact to the world through the use of technology
              </p>
              <p>
                The team consists of
              </p>
              <ul>
                <li>Emon Sarker</li>
                <li>Parinda Rahman</li>
                <li>Mahima Ahsan</li>
              </ul>
            </div>  
        </Grid>
      </Grid>
      </div>
    </div>
</>
  
  );
}

export default App;
